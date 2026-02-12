# src/vanilla_rag.py
from __future__ import annotations

import sys
import re
from typing import List, Tuple, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import prompts
try:
    from prompts import VANILLA_TEMPLATE
except ImportError:
    VANILLA_TEMPLATE = """Use the following context to answer the question.

CONTEXT:
{context}

QUESTION:
{question}
"""

QUESTION_MARKERS = [
    r"\bQUESTION:\s*",
    r"\bNew question:\s*",
]


# ----------------------------
# Input parsing
# ----------------------------
def _extract_last_question(stdin_text: str) -> str:
    """
    Your app sends a multi-line block (policy + history + QUESTION: ...).
    Extract ONLY the final user question after the last marker.
    """
    t = (stdin_text or "").strip()

    if "\n" not in t and len(t) < 400:
        return t.strip()

    last_pos = -1
    last_marker_len = 0

    for pat in QUESTION_MARKERS:
        matches = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if matches:
            pos = matches[-1].start()
            if pos > last_pos:
                last_pos = pos
                last_marker_len = len(matches[-1].group(0))

    if last_pos >= 0:
        q = t[last_pos + last_marker_len :].strip()
    else:
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        q = lines[-1] if lines else ""

    q = re.sub(r"\s+", " ", q).strip()
    return q


# ----------------------------
# Retrieval helpers
# ----------------------------
def _cap(text: str, max_chars: int = 8000) -> str:
    return (text or "")[:max_chars]


def _base_url(url: str) -> str:
    return (url or "").split("#")[0].strip()


def _is_medline_url(url: str) -> bool:
    u = (url or "").lower()
    return u.startswith("http") and "medlineplus.gov" in u


def _group_best_chunk_per_url(
    results: List[Tuple[Document, float]]
) -> List[Tuple[Document, float, str]]:
    """
    For each base URL, keep ONLY the best (lowest) score chunk.
    Returns list of (doc, score, base_url) sorted by score asc.
    """
    best: Dict[str, Tuple[Document, float]] = {}

    for d, s in results:
        url = _base_url((d.metadata.get("source") or "").strip())
        if not url or not _is_medline_url(url):
            continue
        if url not in best or s < best[url][1]:
            best[url] = (d, s)

    grouped = [(doc, score, url) for url, (doc, score) in best.items()]
    grouped.sort(key=lambda x: x[1])  # lower distance = better similarity
    return grouped


def _select_top_k_relevant_unique(
    grouped: List[Tuple[Document, float, str]],
    k_unique: int = 5,
) -> List[Tuple[Document, float, str]]:
    
    if not grouped:
        return []

    best_score = grouped[0][1]
    margins = [0.15, 0.30, 0.50, 0.80, 1.20, 2.00, 3.00]

    for m in margins:
        gate = best_score + m
        picked = [t for t in grouped if t[1] <= gate][:k_unique]
        if len(picked) >= k_unique:
            return picked

    return grouped[:k_unique]


def _build_context(
    picked: List[Tuple[Document, float, str]],
    max_chars: int = 8000,
    per_source_chars: int = 1400,
) -> str:
    
    parts = []
    for d, s, url in picked:
        text = (d.page_content or "").strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text[:per_source_chars]
        parts.append(f"SOURCE: {url}\n{text}")
    return _cap("\n\n".join(parts), max_chars=max_chars)


# ----------------------------
# Output formatting (NO blank lines like Agentic)
# ----------------------------
def _tighten_answer(text: str) -> str:
    
    t = (text or "").replace("\r\n", "\n")

    # strip trailing whitespace per line
    t = "\n".join(line.rstrip() for line in t.splitlines())

    # collapse 3+ newlines => 2 newlines
    t = re.sub(r"\n{3,}", "\n\n", t)

    # remove blank lines immediately after a heading like "1) Overview"
    t = re.sub(r"(?m)^(\d+\)\s[^\n]+)\n\s*\n+", r"\1\n", t)

    # remove blank lines BEFORE bullet points
    t = re.sub(r"\n\s*\n(\s*[-•]\s+)", r"\n\1", t)

    # remove blank lines BETWEEN bullet points
    t = re.sub(r"(?m)^(\s*[-•]\s+.*)\n\s*\n(\s*[-•]\s+)", r"\1\n\2", t)

    # make sure sections are separated by exactly ONE blank line max
    # (i.e., one empty line = "\n\n" is okay, but not more)
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()


# ----------------------------
# LLM answer
# ----------------------------
def generate_structured_answer(llm: ChatOpenAI, question: str, context: str) -> str:
    if not context.strip():
        return _tighten_answer(
            "1) Overview\nNot enough information in the retrieved pages.\n"
            "2) Causes / Risk Factors\nNot enough information in the retrieved pages.\n"
            "3) Symptoms\nNot enough information in the retrieved pages.\n"
            "4) Diagnosis\nNot enough information in the retrieved pages.\n"
            "5) Treatment / What You Can Do\nNot enough information in the retrieved pages.\n"
            "6) When to Seek Urgent Care\nNot enough information in the retrieved pages.\n"
        )

    prompt = VANILLA_TEMPLATE.format(context=context, question=question) + """

CRITICAL REQUIREMENTS:
- Use ONLY the provided MedlinePlus context above.
- Use these headings EXACTLY:

1) Overview
2) Causes / Risk Factors
3) Symptoms
4) Diagnosis
5) Treatment / What You Can Do
6) When to Seek Urgent Care

FORMATTING RULES (VERY IMPORTANT):
- Do NOT add blank lines between a heading and its content.
- Do NOT add blank lines between bullet points.
- If a section is missing in context, write exactly:
  Not enough information in the retrieved pages.
- Do NOT include URLs in the answer body.
- Do NOT use outside knowledge.

Now answer the question:
"""

    out = llm.invoke(prompt).content
    return _tighten_answer(out)


def format_sources_output(picked: List[Tuple[Document, float, str]]) -> str:
    if not picked:
        return "\nSources used (5 unique)\n(no sources found)"

    lines = ["\nSources used (5 unique)"]
    for i, (_, s, url) in enumerate(picked[:5], 1):
        lines.append(f"{i}. {url}  (score={s:.3f})")
    return "\n".join(lines)


def main():
    # Ensure UTF-8 output
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=emb)

    raw_in = sys.stdin.read()
    question = _extract_last_question(raw_in)

    if not question:
        print("ERROR: No question provided", file=sys.stderr)
        sys.exit(1)

    # Retrieve big pool
    raw1 = db.similarity_search_with_score(question, k=220)
    grouped1 = _group_best_chunk_per_url(raw1)
    picked = _select_top_k_relevant_unique(grouped1, k_unique=5)

    # If still not enough, do a light expansion
    if len(picked) < 5:
        expanded = f"{question} symptoms causes diagnosis treatment"
        raw2 = db.similarity_search_with_score(expanded, k=260)
        grouped2 = _group_best_chunk_per_url(raw2)

        merged: Dict[str, Tuple[Document, float, str]] = {}
        for d, s, u in (grouped1 + grouped2):
            if u not in merged or s < merged[u][1]:
                merged[u] = (d, s, u)

        merged_list = list(merged.values())
        merged_list.sort(key=lambda x: x[1])
        picked = _select_top_k_relevant_unique(merged_list, k_unique=5)

    context = _build_context(picked, max_chars=8000, per_source_chars=1400)
    answer = generate_structured_answer(llm, question, context)

    print("\nAnswer\n")
    print(answer)
    print(format_sources_output(picked))


if __name__ == "__main__":
    main()
