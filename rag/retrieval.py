"""
ClearPath Chatbot – BM25 Retrieval Layer
==========================================
Loads the pre-built BM25 index from disk and exposes a single
function to retrieve the top-K most relevant chunks for a query.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import TOP_K_CHUNKS
from rag.database import load_bm25_index, load_chunks
from rag.ingest import tokenize                         


_bm25 = None
_chunks: Optional[List[Dict[str, Any]]] = None


def _ensure_loaded() -> None:
    global _bm25, _chunks
    if _bm25 is not None and _chunks is not None:
        return

    bm25_obj, _ = load_bm25_index()
    chunks = load_chunks()

    if bm25_obj is None or chunks is None:
        raise RuntimeError(
            "BM25 index not found.  Run `python -m rag.ingest` first."
        )

    _bm25 = bm25_obj
    _chunks = chunks


def retrieve(
    query: str,
    top_k: int = TOP_K_CHUNKS,
) -> List[Dict[str, Any]]:
    
    _ensure_loaded()
    assert _bm25 is not None and _chunks is not None

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = _bm25.get_scores(query_tokens)

    max_score = float(np.max(scores)) if np.max(scores) > 0 else 1.0
    normalised = scores / max_score

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        idx = int(idx)
        score = float(normalised[idx])
        if score <= 0:
            continue                        
        chunk = dict(_chunks[idx])         
        chunk["relevance_score"] = round(score, 4)
        results.append(chunk)

    return results


def reload_index() -> None:
    global _bm25, _chunks
    _bm25 = None
    _chunks = None
    _ensure_loaded()


if __name__ == "__main__":
    query = "What is the price of the Pro plan?"
    print(f"Query: {query}\n")
    results = retrieve(query)
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r['relevance_score']:.4f}  "
              f"src={r['source']} p{r['page']}")
        print(f"      {r['text'][:120]}…\n")
