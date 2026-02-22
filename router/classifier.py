"""
ClearPath Chatbot – Deterministic Query Router
================================================
Rules (heuristic mix):
  • A query is "complex" if ANY of the following hold:
      1. Word count ≥ 12
      2. Contains comparison/analytical keywords
         (compare, difference, explain, why, how does, versus, vs,
          pros and cons, trade-off, recommend, evaluate, analyse)
      3. Contains multiple question marks (multi-part question)
      4. Contains subordinate-clause markers
         (because, although, however, whereas, if … then)
  • Otherwise the query is "simple".

Routing:
  • "simple"  → llama-3.1-8b-instant
  • "complex" → llama-3.3-70b-versatile
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_SIMPLE, MODEL_COMPLEX, ROUTER_LOG_FILE, LOG_DIR



_COMPLEX_KEYWORDS = re.compile(
    r"\b("
    r"compare|comparison|difference|differences|versus|vs|"
    r"explain|elaborate|why|how does|how do|how can|how should|"
    r"pros and cons|trade-?off|recommend|evaluate|analyse|analyze|"
    r"step by step|in detail|describe the process|walk me through|"
    r"what are the implications|advantages|disadvantages"
    r")\b",
    re.IGNORECASE,
)

_SUBORDINATE_MARKERS = re.compile(
    r"\b(because|although|however|whereas|nevertheless|"
    r"furthermore|moreover|consequently|if .{3,} then)\b",
    re.IGNORECASE,
)



def classify_query(query: str) -> str:

    words = query.split()
    word_count = len(words)

    if word_count >= 12:
        return "complex"


    if _COMPLEX_KEYWORDS.search(query):
        return "complex"

    if query.count("?") >= 2:
        return "complex"


    if _SUBORDINATE_MARKERS.search(query):
        return "complex"

    return "simple"


def route_query(query: str) -> Dict[str, str]:
 
    classification = classify_query(query)
    model = MODEL_SIMPLE if classification == "simple" else MODEL_COMPLEX
    return {"classification": classification, "model": model}



def log_request(
    query: str,
    classification: str,
    model_used: str,
    tokens_input: int,
    tokens_output: int,
    latency_ms: int,
) -> None:
    
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entry: Dict[str, Any] = {
            "query": query,
            "classification": classification,
            "model_used": model_used,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency_ms": latency_ms,
        }
        with open(ROUTER_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass

if __name__ == "__main__":
    tests = [
        "What is the price?",
        "How do I reset my password?",
        "Compare the Pro and Enterprise plans and explain the differences in features and pricing",
        "What integrations are available? And how do I set them up?",
        "Keyboard shortcuts",
    ]
    for q in tests:
        r = route_query(q)
        print(f"  [{r['classification']:7s}] {r['model'][:20]:20s}  ← \"{q}\"")
