"""
RAG — Retrieval-Augmented Generation Knowledge Search
======================================================
Upgraded from TF-IDF keyword matching to OpenAI text-embedding-3-small
semantic vector search.

WHY THE UPGRADE:
  TF-IDF (old): counts word overlap → fails when interviewers paraphrase.
    "Walk me through your background" → never matched "profile.txt" (no shared words)

  Semantic (new): converts text into meaning-vectors → matches by concept.
    "Walk me through your background" == "Tell me about yourself" == profile
    "Language models" == "LLMs"
    "ML frameworks" == "Machine Learning"

BEHAVIOUR:
  - On first load: embeds all knowledge chunks, caches to ~/.ai_copilot_embeddings.json
  - On subsequent loads: reads cache (instant); re-embeds only if files changed
  - If OpenAI unavailable: falls back to TF-IDF (no crash, no dead air)
  - Sends ONLY the top-k relevant chunks to GPT — NOT the entire knowledge base
"""

import os
import json
import hashlib
import time
from typing import List, Dict, Any, Optional

import numpy as np
import re

# Lazy loader for sklearn to prevent early OpenMP runtime collision with onnxruntime
_TFIDF_VECTORIZER_CLS = None
_SKLEARN_COSINE_SIM_FN = None


def _get_sklearn_components():
    global _TFIDF_VECTORIZER_CLS, _SKLEARN_COSINE_SIM_FN
    if _TFIDF_VECTORIZER_CLS is None:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
            _TFIDF_VECTORIZER_CLS = TfidfVectorizer
            _SKLEARN_COSINE_SIM_FN = sklearn_cosine_similarity
        except Exception:
            return None, None
    return _TFIDF_VECTORIZER_CLS, _SKLEARN_COSINE_SIM_FN

# Load .env so OPENAI_API_KEY is available even when rag.py is imported directly
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ============================================================
# CONSTANTS
# ============================================================

KNOWLEDGE_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "knowledge"))
EMBEDDING_CACHE_PATH = os.path.expanduser("~/.ai_copilot_embeddings.json")
EMBEDDING_MODEL = "text-embedding-3-small"   # $0.02 / 1M tokens ~ $0 per session

_DOCUMENTS_CACHE: List[Dict[str, Any]] = []
_CHUNKS_CACHE: List[Dict[str, Any]] = []       # [{text, filename, embedding, ...}]
_LAST_MTIME: float = 0.0
_openai_client = None                          # lazy-initialised once


def clear_embedding_cache() -> bool:
    """Safely purges the persistent embedding cache file for privacy/PII control."""
    try:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            os.remove(EMBEDDING_CACHE_PATH)
            return True
    except Exception as e:
        print(f"[RAG] Error clearing embedding cache: {e}")
    return False


def get_knowledge_mtime() -> float:
    """Computes max modification time across knowledge folder and all files within it."""
    if not os.path.exists(KNOWLEDGE_FOLDER):
        return 0.0
    latest = os.path.getmtime(KNOWLEDGE_FOLDER)
    try:
        for root, _, files in os.walk(KNOWLEDGE_FOLDER):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    mt = os.path.getmtime(fp)
                    if mt > latest:
                        latest = mt
                except OSError:
                    pass
    except Exception:
        pass
    return latest


# ============================================================
# LOAD KNOWLEDGE DOCUMENTS (WITH IN-MEMORY CACHING)
# ============================================================

def load_documents():
    global _DOCUMENTS_CACHE, _CHUNKS_CACHE, _LAST_MTIME

    if not os.path.exists(KNOWLEDGE_FOLDER):
        return []

    try:
        current_mtime = get_knowledge_mtime()
        if _DOCUMENTS_CACHE and current_mtime == _LAST_MTIME:
            return _DOCUMENTS_CACHE
        _LAST_MTIME = current_mtime
    except Exception:
        pass

    documents = []

    for filename in os.listdir(KNOWLEDGE_FOLDER):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".txt", ".md", ".pdf", ".docx"]:
            continue

        path = os.path.join(
            KNOWLEDGE_FOLDER,
            filename
        )

        text = ""
        try:
            if ext in [".txt", ".md"]:
                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as file:
                    text = file.read().strip()
            elif ext == ".pdf":
                try:
                    import importlib
                    pypdf_mod = importlib.import_module("pypdf")
                    reader = pypdf_mod.PdfReader(path)
                    pages = [p.extract_text() for p in reader.pages if p.extract_text()]
                    text = "\n\n".join(pages).strip()
                except Exception as pe:
                    print(f"Error reading PDF {filename}: {pe}")
            elif ext == ".docx":
                try:
                    import importlib
                    docx_mod = importlib.import_module("docx")
                    doc = docx_mod.Document(path)
                    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                    text = "\n".join(paragraphs).strip()
                except Exception as de:
                    print(f"Error reading DOCX {filename}: {de}")

        except Exception as e:
            print(
                f"Error reading {filename}: {e}"
            )
            continue

        if text:
            documents.append(
                {
                    "filename": filename,
                    "text": text
                }
            )

    _DOCUMENTS_CACHE = documents
    return documents


# ============================================================
# OPENAI CLIENT (lazy init)
# ============================================================

def _get_openai_client():
    """Return a cached OpenAI client, or None if unavailable."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        import importlib
        openai_mod = importlib.import_module("openai")
        _openai_client = openai_mod.OpenAI(api_key=api_key)
        return _openai_client
    except Exception as e:
        print(f"[RAG] OpenAI client unavailable: {e}")
        return None


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def _embed_texts(texts: List[str]) -> Optional[List[List[float]]]:
    """
    Embed a list of texts using OpenAI text-embedding-3-small.
    Cost: ~$0.000002 per chunk (negligible).
    """
    client = _get_openai_client()
    if client is None or not texts:
        return None
    try:
        safe_texts = [t[:6000] for t in texts]  # stay within 8191 token limit
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=safe_texts,
            timeout=2.0
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"[RAG] Embedding API error (falling back to TF-IDF/lexical): {e}")
        return None


def _cosine_sim(vec_a: List[float], vec_b: List[float]) -> float:
    """Fast cosine similarity between two float vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# DISK EMBEDDING CACHE
# ============================================================

def _hash_text(text: str) -> str:
    """Short SHA-256 hash used as cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_embedding_cache() -> Dict[str, List[float]]:
    """Load {text_hash -> embedding} from disk."""
    try:
        if os.path.exists(EMBEDDING_CACHE_PATH):
            with open(EMBEDDING_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_embedding_cache(cache: Dict[str, List[float]]) -> None:
    """Persist {text_hash -> embedding} to disk."""
    try:
        with open(EMBEDDING_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"[RAG] Cache save error: {e}")


# ============================================================
# CHUNKING (Sliding Window Overlap & Semantic Metadata)
# ============================================================

def create_chunks(text: str, max_chars: int = 400, overlap_chars: int = 60) -> List[str]:
    """
    Split text into semantically focused chunks with sliding window overlap.
    - Splits on double newlines (section breaks) first
    - Breaks oversized sections by sentence so each embedding is precise
    - Caps chunks at max_chars to keep embeddings tightly focused
    - Preserves overlap_chars between successive chunk boundaries so split context is retained
    - Preserves 100% backward compatibility returning List[str]
    """
    if not text:
        return []
        
    chunks: List[str] = []
    for section in text.split("\n\n"):
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            sentences = section.replace(". ", ".\n").split("\n")
            current = ""
            overlap_prefix = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                candidate = (current + " " + sentence).strip() if current else sentence
                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                        if overlap_chars > 0 and len(current) > overlap_chars:
                            tail = current[-overlap_chars:]
                            space_idx = tail.find(" ")
                            if space_idx != -1 and space_idx < len(tail) - 5:
                                overlap_prefix = tail[space_idx + 1:].strip()
                            else:
                                overlap_prefix = tail.strip()
                        else:
                            overlap_prefix = ""
                    current = (overlap_prefix + " " + sentence).strip() if overlap_prefix else sentence
            if current and (not chunks or current != chunks[-1]):
                chunks.append(current)
    return chunks


def _extract_section_header(text: str, pos: int) -> str:
    """Finds the nearest preceding section header or uppercase title in text before pos."""
    preceding = text[:pos]
    # Look for markdown header or uppercase section title
    header_patterns = [
        r'(?:^|\n)#{1,4}\s+([^\n]+)',
        r'(?:^|\n)==+\s*([^\n=]+)\s*==+',
        r'(?:^|\n)([A-Z][A-Z\s/&-]{2,30}):(?:\s|\n|$)',
        r'(?:^|\n)([A-Z\s]{4,30})(?:\n|$)'
    ]
    latest_header = ""
    latest_idx = -1
    for pat in header_patterns:
        for m in re.finditer(pat, preceding):
            if m.start() > latest_idx:
                latest_idx = m.start()
                latest_header = m.group(1).strip()
    return latest_header


def create_semantic_chunks(
    text: str,
    max_chars: int = 400,
    overlap_chars: int = 60,
    filename: str = ""
) -> List[Dict[str, Any]]:
    """
    Creates structured chunks containing text, character boundaries, section headers,
    and estimated token counts for enriched citation and precision retrieval.
    """
    if not text:
        return []

    raw_chunks = create_chunks(text, max_chars=max_chars, overlap_chars=overlap_chars)
    structured: List[Dict[str, Any]] = []
    
    current_search_idx = 0
    clean_filename = os.path.basename(filename) if filename else "document"
    doc_stem = os.path.splitext(clean_filename)[0]

    for idx, chunk_text in enumerate(raw_chunks):
        # Locate chunk position in original text
        char_start = text.find(chunk_text[:min(40, len(chunk_text))], current_search_idx)
        if char_start == -1:
            char_start = current_search_idx
        char_end = char_start + len(chunk_text)
        current_search_idx = max(char_start, current_search_idx)

        header = _extract_section_header(text, char_start)
        h_hash = _hash_text(chunk_text)[:8]
        chunk_id = f"{doc_stem}_c{idx}_{h_hash}"
        token_est = max(1, len(chunk_text) // 4)

        structured.append({
            "chunk_id": chunk_id,
            "text": chunk_text,
            "filename": clean_filename,
            "char_start": char_start,
            "char_end": char_end,
            "section_header": header,
            "token_estimate": token_est
        })

    return structured


# ============================================================
# BUILD EMBEDDED CHUNK INDEX (with disk cache & metadata)
# ============================================================

def _build_chunk_index(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts all document chunks into embedded vectors with rich metadata.
    Uses disk cache so chunks are only re-embedded when
    their content changes — saves API cost and latency.
    """
    global _CHUNKS_CACHE

    raw_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        doc_filename = doc.get("filename", "")
        doc_structured = create_semantic_chunks(
            doc.get("text", ""),
            max_chars=400,
            overlap_chars=60,
            filename=doc_filename
        )
        raw_chunks.extend(doc_structured)

    if not raw_chunks:
        _CHUNKS_CACHE = []
        return []

    disk_cache = _load_embedding_cache()

    # Find which chunks are not in the cache yet
    chunks_needing_embed = []
    for i, chunk in enumerate(raw_chunks):
        h = _hash_text(chunk["text"])
        chunk["_hash"] = h
        if h not in disk_cache:
            chunks_needing_embed.append((i, chunk["text"]))

    # Batch-embed only uncached chunks (single API call)
    if chunks_needing_embed:
        texts_to_embed = [t for _, t in chunks_needing_embed]
        print(f"[RAG] Embedding {len(texts_to_embed)} new chunk(s) via OpenAI...")
        t0 = time.perf_counter()
        vectors = _embed_texts(texts_to_embed)
        if vectors:
            for (idx, _), vec in zip(chunks_needing_embed, vectors):
                h = raw_chunks[idx]["_hash"]
                disk_cache[h] = vec
            _save_embedding_cache(disk_cache)
            print(f"[RAG] Embedding done in {(time.perf_counter()-t0)*1000:.0f}ms, cache saved.")
        else:
            print("[RAG] OpenAI embedding failed — TF-IDF fallback will be used.")

    # Attach embeddings (None for chunks that couldn't be embedded)
    for chunk in raw_chunks:
        chunk["embedding"] = disk_cache.get(chunk["_hash"])

    _CHUNKS_CACHE = raw_chunks
    return raw_chunks


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def _semantic_search(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """Embed the question and rank chunks by cosine similarity of meaning."""
    embedded_chunks = [c for c in chunks if c.get("embedding")]
    if not embedded_chunks:
        return []

    query_vecs = _embed_texts([query])
    if not query_vecs:
        return []

    query_vec = query_vecs[0]
    scored = []
    for c in embedded_chunks:
        sim = _cosine_sim(query_vec, c["embedding"])
        if sim > 0.25:
            c_copy = dict(c)
            c_copy["semantic_score"] = sim
            scored.append((sim, c_copy))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


# ============================================================
# TF-IDF SEARCH
# ============================================================

def _tfidf_search(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int
) -> List[Dict[str, Any]]:
    """Keyword-based TF-IDF search fallback or hybrid component."""
    if not chunks:
        return []
    tfidf_cls, cosine_sim_fn = _get_sklearn_components()
    if not tfidf_cls or not cosine_sim_fn:
        return []
    texts = [c["text"] for c in chunks]
    try:
        vectorizer = tfidf_cls(stop_words="english", ngram_range=(1, 2))
        vectors = vectorizer.fit_transform(texts + [query])
        similarities = cosine_sim_fn(vectors[-1], vectors[:-1])[0]
        ranked = similarities.argsort()[::-1]
        results = []
        for i in ranked[:top_k]:
            sim = float(similarities[i])
            if sim > 0:
                c_copy = dict(chunks[i])
                c_copy["tfidf_score"] = sim
                results.append(c_copy)
        return results
    except Exception as e:
        print(f"[RAG] TF-IDF search error: {e}")
        return []


# ============================================================
# STRUCTURED HYBRID SEARCH WITH RECIPROCAL RANK FUSION (RRF)
# ============================================================

def search_knowledge_structured(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve structured candidate knowledge chunks ranked with Reciprocal Rank Fusion (RRF).
    Combines dense semantic vector search with sparse TF-IDF keyword retrieval:
        RRF_score = (1.0 / (60 + sem_rank)) + (0.8 / (60 + tfidf_rank))
    """
    if not question or not question.strip():
        documents = load_documents()
        if not documents:
            return []
        raw_chunks = _build_chunk_index(documents)
        return raw_chunks[:top_k]

    documents = load_documents()
    if not documents:
        return []

    chunks = _build_chunk_index(documents)
    if not chunks:
        return []

    # Enrich query with recent conversation turns
    history_context = ""
    if conversation_history:
        recent = conversation_history[-4:]
        parts = [m.get("content", "") for m in recent if m.get("content")]
        history_context = "\n".join(parts)

    retrieval_query = f"{question}\n\n{history_context}".strip()

    # 1. Dense Semantic Search (retrieve 2x top_k for fusion)
    has_embeddings = any(c.get("embedding") for c in chunks)
    sem_results = _semantic_search(retrieval_query, chunks, top_k=top_k * 2) if has_embeddings else []

    # 2. Sparse TF-IDF Search (retrieve 2x top_k for fusion)
    tfidf_results = _tfidf_search(retrieval_query, chunks, top_k=top_k * 2)

    # 3. Reciprocal Rank Fusion
    if sem_results and tfidf_results:
        rrf_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(sem_results):
            cid = item.get("chunk_id") or _hash_text(item["text"])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank + 1.0))
            chunk_lookup[cid] = item

        for rank, item in enumerate(tfidf_results):
            cid = item.get("chunk_id") or _hash_text(item["text"])
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (0.8 / (60.0 + rank + 1.0))
            if cid not in chunk_lookup:
                chunk_lookup[cid] = item

        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
        fused: List[Dict[str, Any]] = []
        for cid in sorted_cids[:top_k]:
            item = dict(chunk_lookup[cid])
            item["score"] = rrf_scores[cid]
            item["method"] = "HYBRID_RRF"
            fused.append(item)
        return fused

    elif sem_results:
        for item in sem_results:
            item["score"] = item.get("semantic_score", 0.0)
            item["method"] = "SEMANTIC"
        return sem_results[:top_k]

    elif tfidf_results:
        for item in tfidf_results:
            item["score"] = item.get("tfidf_score", 0.0)
            item["method"] = "TFIDF"
        return tfidf_results[:top_k]

    return []


# ============================================================
# MAIN PUBLIC FUNCTION — search_knowledge()
# ============================================================

def search_knowledge(
    question: str,
    conversation_history: Optional[List[Dict]] = None,
    top_k: int = 5
) -> str:
    """
    Retrieve the most semantically relevant candidate knowledge for a question.
    Upgraded with Reciprocal Rank Fusion (RRF) and precise source citations.

    Args:
        question:             The interview question text
        conversation_history: Recent conversation turns for follow-up context
        top_k:                Max number of chunks to return (default 5)

    Returns:
        A formatted string of the most relevant candidate facts with source citations.
    """
    # Guard: empty question causes OpenAI API error — return all docs as fallback
    if not question or not question.strip():
        documents = load_documents()
        if not documents:
            return ""
        all_text = "\n\n".join(
            f"--- {doc['filename']} ---\n{doc['text']}"
            for doc in documents
        )
        return f"==== CANDIDATE INFORMATION ====\n{all_text}\n"

    documents = load_documents()
    if not documents:
        return ""

    relevant_chunks = search_knowledge_structured(
        question=question,
        conversation_history=conversation_history,
        top_k=top_k
    )

    # Last resort fallback: return all docs if nothing matched
    if not relevant_chunks:
        all_text = "\n\n".join(
            f"--- {doc['filename']} ---\n{doc['text']}"
            for doc in documents
        )
        return f"==== CANDIDATE INFORMATION ====\n{all_text}\n"

    search_method = relevant_chunks[0].get("method", "HYBRID")

    # Build clean context string with source citations
    formatted_chunks = []
    for chunk in relevant_chunks:
        fname = chunk.get("filename", "knowledge")
        score = chunk.get("score")
        header = chunk.get("section_header")

        citation_parts = [f"Source: {fname}"]
        if score is not None:
            if chunk.get("method") == "HYBRID_RRF":
                citation_parts.append(f"RRF Score: {score:.3f}")
            else:
                citation_parts.append(f"Match: {score:.0%}")
        if header:
            citation_parts.append(f"Section: {header}")

        citation_tag = f"[{' | '.join(citation_parts)}]"
        formatted_chunks.append(f"{citation_tag}\n{chunk['text']}")

    relevant_section = "\n\n".join(formatted_chunks)

    return f"""==== RELEVANT CANDIDATE INFORMATION ({search_method} MATCH) ====

{relevant_section}

===================================================
"""