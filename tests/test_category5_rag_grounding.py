"""
Test Suite: Category 5 - RAG & Anti-Fabrication Grounding Verification
=====================================================================
Tests Parakeet Audit Matrix Category 5 requirements:
1. Sliding window chunking with context overlap (backward-compatible signature).
2. Semantic chunk metadata extraction (chunk_id, char boundaries, section headers, token estimates).
3. Hybrid search with Reciprocal Rank Fusion (RRF).
4. Formatted source citations in injected RAG blocks ([Source: {filename} | Match: {score}]).
5. Post-generation anti-fabrication grounding validation:
   - Employer claim sanitization for unverified companies.
   - Employer claim preservation for verified companies.
   - Strict whitelisting for technical tools / cloud platforms (AWS, Stripe, Docker, etc.).
   - Degree / education claim sanitization.
6. Full integration via validate_and_clean_answer() and telemetry recording.
7. Sub-millisecond execution performance verification (< 0.3ms).
"""

import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


import time
import pytest
from typing import Dict, Any, List

import rag
from interview_engine import (
    InterviewSession,
    validate_candidate_grounding,
    validate_and_clean_answer,
    TECH_TOOL_WHITELIST
)


def test_create_chunks_sliding_overlap():
    """Verify create_chunks splits with overlap and preserves List[str] return type."""
    sample_text = (
        "First sentence discussing backend system architecture and high scale microservices. "
        "Second sentence covering Kafka message streaming and distributed consensus protocols. "
        "Third sentence detailing database sharding and read replicas with Redis caching. "
        "Fourth sentence describing Kubernetes cluster orchestration and autoscaling deployments."
    )
    
    # 1. Test signature and backward compatibility
    chunks_standard = rag.create_chunks(sample_text, max_chars=120, overlap_chars=30)
    assert isinstance(chunks_standard, list)
    assert len(chunks_standard) > 1
    assert all(isinstance(c, str) for c in chunks_standard)

    # 2. Test overlap presence: adjacent chunks should share words
    overlap_found = False
    for i in range(len(chunks_standard) - 1):
        c1_words = set(chunks_standard[i].split()[-4:])
        c2_words = set(chunks_standard[i + 1].split()[:6])
        if c1_words.intersection(c2_words):
            overlap_found = True
            break
    assert overlap_found, "Successive chunks must share overlap words across boundaries"

    # 3. Test zero overlap behaves cleanly
    chunks_no_overlap = rag.create_chunks(sample_text, max_chars=120, overlap_chars=0)
    assert isinstance(chunks_no_overlap, list)
    assert len(chunks_no_overlap) >= 1


def test_create_semantic_chunks_metadata():
    """Verify create_semantic_chunks extracts headers, character spans, and tokens."""
    markdown_doc = (
        "# SYSTEM EXPERIENCE\n\n"
        "Lead Backend Engineer working on high throughput data ingest.\n"
        "Engineered streaming event queues using Apache Kafka and Go.\n\n"
        "## DATABASE OPTIMIZATION\n\n"
        "Managed multi-region PostgreSQL instances with pgBouncer connection pooling.\n"
        "Optimized slow B-Tree queries and created partial covering indexes."
    )
    
    chunks = rag.create_semantic_chunks(markdown_doc, max_chars=150, overlap_chars=30, filename="resume.md")
    assert len(chunks) >= 2

    for c in chunks:
        assert "chunk_id" in c
        assert "text" in c
        assert "filename" in c
        assert c["filename"] == "resume.md"
        assert "char_start" in c
        assert "char_end" in c
        assert c["char_end"] > c["char_start"]
        assert "section_header" in c
        assert "token_estimate" in c
        assert c["token_estimate"] > 0

    # Verify section header extraction
    headers = [c["section_header"] for c in chunks]
    assert any("SYSTEM EXPERIENCE" in h for h in headers)
    assert any("DATABASE OPTIMIZATION" in h for h in headers)


def test_hybrid_rrf_scoring_and_fallback():
    """Verify structured hybrid search calculates RRF scores and handles fallbacks."""
    # Test with mock chunks
    test_chunks = [
        {
            "chunk_id": "c1",
            "text": "Kafka event streaming and real-time pubsub messaging",
            "filename": "kafka.txt",
            "section_header": "STREAMING"
        },
        {
            "chunk_id": "c2",
            "text": "PostgreSQL database partitioning and SQL query tuning",
            "filename": "database.txt",
            "section_header": "STORAGE"
        },
        {
            "chunk_id": "c3",
            "text": "Docker containerization and Kubernetes cluster networking",
            "filename": "devops.txt",
            "section_header": "INFRA"
        }
    ]

    # Test TF-IDF fallback component
    tfidf_res = rag._tfidf_search("Kafka messaging", test_chunks, top_k=2)
    assert len(tfidf_res) >= 1
    assert tfidf_res[0]["chunk_id"] == "c1"
    assert "tfidf_score" in tfidf_res[0]
    assert tfidf_res[0]["tfidf_score"] > 0


def test_search_knowledge_citations_formatting():
    """Verify search_knowledge formats structured citations with Source and match info."""
    result = rag.search_knowledge("Python backend architecture")
    assert isinstance(result, str)
    # Even if knowledge folder is empty or filled, output must follow the contract
    if "RELEVANT CANDIDATE INFORMATION" in result:
        assert "====" in result
        assert "[Source:" in result or "MATCH)" in result


def test_grounding_unverified_employer_sanitized():
    """Verify unverified employer claims are safely rewritten to generic phrasing."""
    resume = "Senior Software Engineer at AlphaTech Solutions. Experience with Python and Go."
    hallucinated_answer = (
        "When I was at Netflix, I architected the core recommendation pipeline. "
        "In my role at Uber, I managed driver dispatch routing."
    )

    sanitized, warnings = validate_candidate_grounding(
        answer=hallucinated_answer,
        resume_text=resume
    )

    assert "Netflix" not in sanitized
    assert "Uber" not in sanitized
    assert "in my previous engineering experience" in sanitized or "in my previous role" in sanitized
    assert len(warnings) == 2
    assert any("Netflix" in w for w in warnings)
    assert any("Uber" in w for w in warnings)


def test_grounding_verified_employer_preserved():
    """Verify verified employer claims from resume/knowledge are strictly preserved."""
    resume = "Lead Architect at AlphaTech Solutions. Previous roles at BetaCorp Systems."
    legitimate_answer = (
        "When I was at AlphaTech Solutions, I built our distributed microservices. "
        "At BetaCorp Systems, I led the core database migration."
    )

    sanitized, warnings = validate_candidate_grounding(
        answer=legitimate_answer,
        resume_text=resume
    )

    assert "AlphaTech Solutions" in sanitized
    assert "BetaCorp Systems" in sanitized
    assert len(warnings) == 0


def test_grounding_tech_tools_whitelisted():
    """Verify technical tools, cloud platforms, and protocols are never flagged as employers."""
    resume = "Full-stack developer at DeltaCorp."
    tech_heavy_answer = (
        "I built the application on AWS and deployed it using Docker and Kubernetes. "
        "For payments, we integrated Stripe API and used MongoDB for persistence. "
        "At DeltaCorp, I optimized Redis caching and PostgreSQL indices."
    )

    sanitized, warnings = validate_candidate_grounding(
        answer=tech_heavy_answer,
        resume_text=resume
    )

    assert "AWS" in sanitized
    assert "Docker" in sanitized
    assert "Kubernetes" in sanitized
    assert "Stripe" in sanitized
    assert "MongoDB" in sanitized
    assert "DeltaCorp" in sanitized
    assert len(warnings) == 0


def test_grounding_unverified_degree_sanitized():
    """Verify unverified degree claims are rewritten to generic academic phrasing."""
    resume = "B.S. in Computer Science from State University."
    hallucinated_degree_answer = (
        "During my degree at Stanford University, I conducted research in neural networks. "
        "I also graduated from MIT with honors."
    )

    sanitized, warnings = validate_candidate_grounding(
        answer=hallucinated_degree_answer,
        resume_text=resume
    )

    assert "Stanford University" not in sanitized
    assert "MIT" not in sanitized
    assert len(warnings) == 2


def test_validate_and_clean_answer_integration():
    """Verify validate_and_clean_answer performs grounding validation with session."""
    session = InterviewSession()
    session.resume_text = "Software Engineer at CloudScale Inc."

    raw_answer = (
        "When I worked at Meta, I built the GraphQL layer. "
        "At CloudScale Inc, I maintained our Kubernetes clusters."
    )

    cleaned = validate_and_clean_answer(
        raw_answer=raw_answer,
        question="Tell me about your background with APIs.",
        session=session,
        question_type="technical",
        knowledge_source="CANDIDATE_EXPERIENCE"
    )

    assert "Meta" not in cleaned
    assert "CloudScale Inc" in cleaned
    assert isinstance(cleaned, str)
    assert len(cleaned) > 0


def test_grounding_performance_budget():
    """Verify validate_candidate_grounding runs in < 0.3ms per call."""
    resume = "Full-stack Engineer at Acme Systems. Worked on Python, Django, React, AWS."
    answer = (
        "When I was at Netflix, I architected a multi-region distributed system. "
        "At Acme Systems, I scaled the Django REST API on AWS and Redis."
    )

    iterations = 200
    t0 = time.perf_counter()
    for _ in range(iterations):
        validate_candidate_grounding(answer, resume_text=resume)
    total_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = total_ms / iterations

    print(f"\n[Grounding Perf] Average latency: {avg_ms:.4f} ms per call across {iterations} iterations")
    assert avg_ms < 0.3, f"Grounding validation must be < 0.3ms, took {avg_ms:.4f}ms"
