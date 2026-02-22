"""
ClearPath Chatbot – Chunking Layer
===================================
Responsibilities:
  1. Extract raw text from each PDF using PyMuPDF (fitz).
  2. Split the text into word-level chunks of ~CHUNK_SIZE_WORDS with
     CHUNK_OVERLAP_WORDS overlap.
  3. Return a list of chunk dicts:
       {
         "text": "...",
         "source": "14_Pricing_Sheet_2024.pdf",
         "page": 2,
         "chunk_id": 42
       }
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any

import fitz 

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS, PDF_DIR



def _clean_text(raw: str) -> str:
    text = raw.replace("\x00", "")
    text = re.sub(r"[^\S\n]+", " ", text)      
    text = re.sub(r"\n{3,}", "\n\n", text)       
    return text.strip()


def _tokenize_words(text: str) -> List[str]:
    return text.split()



def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
   
    pages: List[Dict[str, Any]] = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean_text(raw)
        if cleaned:
            pages.append({"page": page_num, "text": cleaned})
    doc.close()
    return pages


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_filename: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[Dict[str, Any]]:
   
    words: List[str] = []
    word_page_map: List[int] = [] 

    for p in pages:
        page_words = _tokenize_words(p["text"])
        words.extend(page_words)
        word_page_map.extend([p["page"]] * len(page_words))

    if not words:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]

        page_counts: Dict[int, int] = {}
        for idx in range(start, end):
            pg = word_page_map[idx]
            page_counts[pg] = page_counts.get(pg, 0) + 1
        majority_page = max(page_counts, key=page_counts.get) 

        chunks.append({
            "text": " ".join(chunk_words),
            "source": source_filename,
            "page": majority_page,
        })

        start += chunk_size - overlap
        if start >= len(words):
            break

    return chunks


def process_all_pdfs(pdf_dir: str = PDF_DIR) -> List[Dict[str, Any]]:
    
    all_chunks: List[Dict[str, Any]] = []
    pdf_files = sorted(
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )

    for filename in pdf_files:
        filepath = os.path.join(pdf_dir, filename)
        pages = extract_text_from_pdf(filepath)
        doc_chunks = chunk_pages(pages, source_filename=filename)
        all_chunks.extend(doc_chunks)
        print(f"  ✓ {filename}: {len(pages)} page(s) → {len(doc_chunks)} chunk(s)")

    for idx, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = idx

    print(f"\n  Total: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")
    return all_chunks


if __name__ == "__main__":
    chunks = process_all_pdfs()
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk #{sample['chunk_id']}:")
        print(f"  Source : {sample['source']} (page {sample['page']})")
        print(f"  Words  : {len(sample['text'].split())}")
        print(f"  Preview: {sample['text'][:200]}…")
