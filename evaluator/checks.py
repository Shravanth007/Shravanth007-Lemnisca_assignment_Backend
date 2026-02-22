"""
ClearPath Chatbot – Output Evaluator
======================================
After the LLM generates an answer, this module inspects the response
and raises informational flags.

Implemented flags
-----------------
1. "no_context"          – LLM answered but 0 chunks were retrieved.
2. "refusal"             – LLM explicitly refused / said it doesn't know.
3. "internal_data_leak"  – Retrieved chunks belong to Internal Policies or
                           Internal Operations docs AND the query was classified
                           as "simple" (a basic customer question should not
                           surface internal HR/ops content).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import INTERNAL_DOC_PREFIXES



_REFUSAL_PATTERNS = re.compile(
    r"("
    r"i don'?t know|i do not know|"
    r"i cannot help|i can'?t help|"
    r"not mentioned in the documentation|"
    r"not mentioned in the provided|"
    r"no information available|"
    r"i don'?t have (enough )?information|"
    r"i do not have (enough )?information|"
    r"cannot find|can'?t find|"
    r"unable to (find|answer|provide)|"
    r"no relevant (information|data|documents)|"
    r"the (documents|documentation) (doesn'?t|does not|do not) (mention|contain|include)"
    r")",
    re.IGNORECASE,
)


def _is_internal_doc(source_filename: str) -> bool:
    for prefix in INTERNAL_DOC_PREFIXES:
        if source_filename.startswith(prefix):
            return True
    return False



def evaluate(
    answer: str,
    chunks_retrieved: int,
    sources: List[Dict[str, Any]],
    classification: str,
) -> List[str]:
    """
    Inspect the LLM answer and context to produce a list of flag strings.

    Parameters
    ----------
    answer : str
        The generated LLM response text.
    chunks_retrieved : int
        Number of chunks that were retrieved (may be 0).
    sources : list[dict]
        The source dicts returned to the caller (each has a ``document`` key).
    classification : str
        The router classification ("simple" or "complex").

    Returns
    -------
    list[str]
        A (possibly empty) list of flag strings.
    """
    flags: List[str] = []

    if chunks_retrieved == 0 and not _REFUSAL_PATTERNS.search(answer):
        flags.append("no_context")

    if _REFUSAL_PATTERNS.search(answer):
        flags.append("refusal")

    if classification == "simple" and sources:
        internal_sources = [
            s for s in sources if _is_internal_doc(s.get("document", ""))
        ]
        if internal_sources:
            flags.append("internal_data_leak")

    return flags
