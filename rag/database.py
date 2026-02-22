"""
ClearPath Chatbot – Database / Persistence Layer
==================================================
Saves and loads:
  • The list of chunk dicts  →  JSON
  • The BM25 index object   →  pickle

Everything is stored under  backend/rag/index_store/
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import INDEX_DIR


CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")
BM25_INDEX_FILE = os.path.join(INDEX_DIR, "bm25_index.pkl")
TOKENIZED_CORPUS_FILE = os.path.join(INDEX_DIR, "tokenized_corpus.pkl")


def _ensure_dir() -> None:
    os.makedirs(INDEX_DIR, exist_ok=True)



def save_chunks(chunks: List[Dict[str, Any]]) -> str:
    _ensure_dir()
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  💾 Saved {len(chunks)} chunks → {CHUNKS_FILE}")
    return CHUNKS_FILE


def load_chunks() -> Optional[List[Dict[str, Any]]]:
    if not os.path.isfile(CHUNKS_FILE):
        return None
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)



def save_bm25_index(bm25_obj: Any, tokenized_corpus: List[List[str]]) -> str:
    _ensure_dir()
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_obj, f)
    with open(TOKENIZED_CORPUS_FILE, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    print(f"  💾 Saved BM25 index → {BM25_INDEX_FILE}")
    return BM25_INDEX_FILE


def load_bm25_index():
   
    if not os.path.isfile(BM25_INDEX_FILE) or not os.path.isfile(TOKENIZED_CORPUS_FILE):
        return None, None
    with open(BM25_INDEX_FILE, "rb") as f:
        bm25_obj = pickle.load(f)
    with open(TOKENIZED_CORPUS_FILE, "rb") as f:
        tokenized_corpus = pickle.load(f)
    return bm25_obj, tokenized_corpus



def index_exists() -> bool:
    return os.path.isfile(CHUNKS_FILE) and os.path.isfile(BM25_INDEX_FILE)
