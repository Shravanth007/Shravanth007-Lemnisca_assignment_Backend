"""
ClearPath Chatbot – Ingestion Pipeline
========================================
Orchestrates:
  1.  PDF extraction + chunking   (chunking.py)
  2.  Tokenisation of every chunk
  3.  BM25 index construction      (rank_bm25)
  4.  Persistence to disk          (database.py)

Run this script once (or whenever the PDFs change) to rebuild the index:

    python -m rag.ingest          # from the backend/ directory
"""

from __future__ import annotations

import os
import re
import sys
import time
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PDF_DIR
from rag.chunking import process_all_pdfs
from rag.database import save_chunks, save_bm25_index, index_exists

from rank_bm25 import BM25Okapi


_STOP_WORDS = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "shall should may might can could of in to for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but or if while about up it its he she they them their this that these those "
    "i me my we our you your am".split()
)


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]



def build_index(force: bool = False) -> None:
    if index_exists() and not force:
        print("⚡ Index already exists. Use --force to rebuild.")
        return

    print("=" * 60)
    print("  ClearPath – Building BM25 Index")
    print("=" * 60)

    t0 = time.perf_counter()

    print("\n📄 Step 1: Extracting & chunking PDFs …")
    chunks = process_all_pdfs(PDF_DIR)

    if not chunks:
        print("❌ No chunks produced – check that PDFs exist in", PDF_DIR)
        return

    print("\n🔤 Step 2: Tokenising chunks …")
    tokenized_corpus: List[List[str]] = [tokenize(c["text"]) for c in chunks]
    avg_tokens = sum(len(t) for t in tokenized_corpus) / len(tokenized_corpus)
    print(f"  Average tokens per chunk: {avg_tokens:.1f}")

    print("\n📊 Step 3: Building BM25Okapi index …")
    bm25 = BM25Okapi(tokenized_corpus)
    print("  ✓ BM25 index built")

    print("\n💾 Step 4: Saving to disk …")
    save_chunks(chunks)
    save_bm25_index(bm25, tokenized_corpus)

    elapsed = time.perf_counter() - t0
    print(f"\n✅ Done in {elapsed:.2f}s  ({len(chunks)} chunks indexed)")



if __name__ == "__main__":
    force = "--force" in sys.argv
    build_index(force=force)
