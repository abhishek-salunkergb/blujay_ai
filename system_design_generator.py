"""
System Design Architecture & Diagram Generator (Feature 6)
==========================================================
Generates high-level system architecture diagrams, ASCII box layouts,
back-of-the-envelope scale calculations, and core trade-offs on-demand
for FAANG / Senior System Design interview rounds.
"""

import os
import re
import json
from typing import Dict, Any, Optional
from groq import Groq


def generate_heuristic_architecture(topic: str) -> Dict[str, Any]:
    """Generates clean ASCII architecture and scale calculations offline."""
    clean_topic = topic.strip() or "Distributed Web Application"
    
    ascii_diagram = f"""
+-----------------------------------------------------------------------------------+
|                            HIGH LEVEL SYSTEM ARCHITECTURE                         |
|                                Target: {clean_topic[:40]}                          |
+-----------------------------------------------------------------------------------+

                          [ Global Users / Mobile / Web ]
                                         |
                                         v
                            [ Cloudflare CDN / Route 53 ]
                                         |
                                         v
                         [ Multi-AZ NLB / API Gateway ]
                          (Rate Limiting & Auth Token)
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
    [ Ingestion / Core Service ]                         [ Query / Read Service ]
      (Auto-Scaling K8s Pods)                              (Auto-Scaling K8s Pods)
               |                                                   |
               +-------------------------+                         |
               |                         |                         v
               v                         v              [ Redis / Memcached Cluster ]
      [ Apache Kafka / RabbitMQ ]  [ Primary DB (Write) ]      (Sub-millisecond Reads)
         (Event-Driven Streaming)    (PostgreSQL / DynamoDB)               |
               |                         |                                 |
               v                         v (Async Replica Sync)            v
    [ Async Consumer Workers ]     [ Read Replicas ] ----------------------+
         (Batch Processing)        (Multi-Region Read)
               |
               v
      [ S3 / Blob Storage ] + [ OpenSearch / Analytics ]
+-----------------------------------------------------------------------------------+
"""

    mermaid_diagram = f"""graph TD
    Client[Global Clients] --> CDN[Cloudflare CDN / DNS]
    CDN --> LB[API Gateway / NLB]
    LB --> CoreService[Core Application Service]
    LB --> ReadService[Read / Query Service]
    ReadService --> Cache[(Redis Cache Cluster)]
    CoreService --> Queue[Kafka / Event Stream]
    CoreService --> PrimaryDB[(Primary DB - Write)]
    PrimaryDB -.-> ReadReplica[(Read Replicas)]
    ReadService --> ReadReplica
    Queue --> Workers[Background Async Workers]
    Workers --> S3[(S3 Object Storage)]
    Workers --> Search[(OpenSearch / Elastic)]
"""

    scale_estimations = {
        "daily_active_users": "10 Million DAU",
        "read_write_ratio": "10:1 (Read-Heavy)",
        "read_qps": "~12,000 QPS (Peak ~25,000 QPS)",
        "write_qps": "~1,200 QPS (Peak ~2,500 QPS)",
        "storage_per_year": "~36 TB / year (with replication ~108 TB)"
    }

    trade_offs = [
        "Eventual Consistency over Strong Consistency (CAP: AP preferred for high availability).",
        "Cache-Aside Pattern with TTL + Write-Through for hot partition mitigation.",
        "Database Horizontal Sharding by User ID hash to prevent single-node hotspotting.",
        "Kafka Dead-Letter Queue (DLQ) for resilient failure handling in async pipelines."
    ]

    talking_points = [
        f"1. Clarify functional requirements, read/write ratios, and latency SLAs for {clean_topic}.",
        "2. Establish the high-level data flow from Client -> Gateway -> Services -> Cache -> DB.",
        "3. Address bottlenecks: rate limiting, caching strategies, horizontal database scaling.",
        "4. Deep-dive into failure modes: replica lag, node failovers, and idempotency keys."
    ]

    return {
        "topic": clean_topic,
        "ascii_diagram": ascii_diagram.strip(),
        "mermaid_diagram": mermaid_diagram.strip(),
        "scale_estimations": scale_estimations,
        "key_trade_offs": trade_offs,
        "interview_talking_points": talking_points
    }


def generate_system_design_diagram(
    topic: str,
    client: Optional[Groq] = None,
    custom_requirements: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates structured system design breakdown with ASCII diagrams, scale math, and trade-offs.
    Uses Groq LLM with offline heuristic fallback.
    """
    if not client:
        return generate_heuristic_architecture(topic)

    prompt = f"""
You are a Staff Distributed Systems Architect and Principal FAANG System Design Interviewer.
Design a robust, scalable architecture for:

SYSTEM TO DESIGN: {topic}
ADDITIONAL REQUIREMENTS: {custom_requirements or 'Scale to 50M DAU with 99.99% availability and low latency.'}

Return a valid JSON object matching this schema:
{{
  "topic": "{topic}",
  "ascii_diagram": "<A clear, beautiful ASCII box diagram showing Clients, Gateway, Services, Cache, Queues, Databases, and Storage>",
  "mermaid_diagram": "<Mermaid graph TD code>",
  "scale_estimations": {{
     "daily_active_users": "<string>",
     "read_write_ratio": "<string>",
     "read_qps": "<string>",
     "write_qps": "<string>",
     "storage_per_year": "<string>"
  }},
  "key_trade_offs": [<list of 3-4 bullet strings explaining architectural trade-offs like CAP, consistency vs latency, caching>],
  "interview_talking_points": [<list of 4 structured step-by-step points candidate should explain during the interview>]
}}
Only return valid JSON without markdown code fence formatting.
"""
    models_to_try = [
        os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "llama-3.3-70b-versatile",
        os.getenv("FAST_GROQ_MODEL", "llama-3.1-8b-instant"),
        "llama-3.1-8b-instant"
    ]
    models_to_try = list(dict.fromkeys(models_to_try))
    for model_name in models_to_try:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1400
            )
            content = completion.choices[0].message.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
            data = json.loads(content)
            if "ascii_diagram" in data and "key_trade_offs" in data:
                return data
        except Exception as e:
            continue

    return generate_heuristic_architecture(topic)
