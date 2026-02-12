# src/agentic_rag.py
from __future__ import annotations

import re
import sys
from typing import List, Tuple, Dict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# ----------------------------
# Question extraction (works with your app’s multi-line stdin)
# ----------------------------
QUESTION_MARKERS = [r"\bQUESTION:\s*", r"\bNew question:\s*"]

def extract_last_question(stdin_text: str) -> str:
    t = (stdin_text or "").strip()
    if "\n" not in t and len(t) < 400:
        return re.sub(r"\s+", " ", t).strip()

    last_pos, last_len = -1, 0
    for pat in QUESTION_MARKERS:
        matches = list(re.finditer(pat, t, flags=re.IGNORECASE))
        if matches:
            m = matches[-1]
            if m.start() > last_pos:
                last_pos, last_len = m.start(), len(m.group(0))

    if last_pos >= 0:
        q = t[last_pos + last_len :].strip()
    else:
        lines = [x.strip() for x in t.splitlines() if x.strip()]
        q = lines[-1] if lines else ""

    return re.sub(r"\s+", " ", q).strip()


# ----------------------------
# URL helpers
# ----------------------------
def base_url(url: str) -> str:
    return (url or "").split("#")[0].strip()

def is_medline(url: str) -> bool:
    return "medlineplus.gov" in (url or "").lower()


# ----------------------------
# 🔒 HARD formatting enforcement: NO blank lines between sections
# ----------------------------
def normalize_answer(ans: str) -> str:
    t = (ans or "").replace("\r\n", "\n")
    t = "\n".join(line.rstrip() for line in t.splitlines())

    # Fix split headings like:
    # 2.
    # Causes ...
    t = re.sub(r"(?m)^\s*(\d+)\s*[\.\)]\s*\n\s*([A-Za-z])", r"\1) \2", t)
    t = re.sub(r"(?m)^\s*(\d+)[\.\)]\s*", r"\1) ", t)

    # Remove blank lines after headings (critical)
    t = re.sub(r"(?m)(\d\)\s[^\n]+)\n\s*\n+", r"\1\n", t)

    # Collapse any remaining extra newlines globally
    t = re.sub(r"\n{2,}", "\n", t)

    return t.strip()


# ----------------------------
# Retrieval helpers
# ----------------------------
def retrieve(db: Chroma, query: str, k: int = 60) -> List[Tuple[Document, float]]:
    try:
        return db.similarity_search_with_score(query, k=k)
    except Exception as e:
        print(f"ERROR: retrieval failed: {e}", file=sys.stderr)
        return []


def group_best_chunk_per_url(
    results: List[Tuple[Document, float]]
) -> List[Tuple[Document, float, str]]:
    """
    Keep only the best (lowest score) chunk per MedlinePlus URL.
    Returns sorted list (best first).
    """
    best: Dict[str, Tuple[Document, float]] = {}
    for d, s in results:
        u = base_url(d.metadata.get("source", ""))
        if not u or not is_medline(u):
            continue
        if u not in best or s < best[u][1]:
            best[u] = (d, s)

    grouped = [(doc, score, url) for url, (doc, score) in best.items()]
    grouped.sort(key=lambda x: x[1])
    return grouped


def select_top_k(grouped: List[Tuple[Document, float, str]], k: int = 5) -> List[Tuple[Document, float, str]]:
    if not grouped:
        return []

    best_score = grouped[0][1]
    margins = [0.15, 0.30, 0.50, 0.80, 1.20, 2.00]

    for m in margins:
        gate = best_score + m
        picked = [x for x in grouped if x[1] <= gate][:k]
        if len(picked) >= min(3, k):
            return picked

    return grouped[:k]


def build_context(picked: List[Tuple[Document, float, str]], max_chars: int = 8000, per_source: int = 1400) -> str:
    blocks = []
    for d, _, u in picked:
        text = re.sub(r"\n{3,}", "\n", (d.page_content or "").strip())
        blocks.append(f"SOURCE: {u}\n{text[:per_source]}")
    return "\n".join(blocks)[:max_chars]


# ----------------------------
# Agentic step: make section-specific queries
# ----------------------------
def section_queries(llm: ChatOpenAI, question: str) -> Dict[str, str]:
    msg = f"""Create 6 short search queries to retrieve MedlinePlus info for this medical question.

Question: {question}

Return EXACTLY 6 lines in this format:
Overview: ...
Causes: ...
Symptoms: ...
Diagnosis: ...
Treatment: ...
Urgent: ...

Rules:
- Each query should be 5–12 words.
- Include key medical terms from the question.
- No extra text.
"""
    try:
        out = llm.invoke(msg).content.strip()
    except Exception:
        out = ""

    mapping: Dict[str, str] = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip().lower()
        val = v.strip()
        if not val:
            continue
        if key.startswith("overview"):
            mapping["Overview"] = val
        elif key.startswith("causes"):
            mapping["Causes"] = val
        elif key.startswith("symptoms"):
            mapping["Symptoms"] = val
        elif key.startswith("diagnosis"):
            mapping["Diagnosis"] = val
        elif key.startswith("treatment"):
            mapping["Treatment"] = val
        elif key.startswith("urgent"):
            mapping["Urgent"] = val

    defaults = {
        "Overview": question,
        "Causes": f"{question} causes risk factors",
        "Symptoms": f"{question} symptoms",
        "Diagnosis": f"{question} diagnosis tests",
        "Treatment": f"{question} treatment management",
        "Urgent": f"{question} emergency when to seek help",
    }
    for k, v in defaults.items():
        mapping.setdefault(k, v)

    return mapping


# ----------------------------
# Answer generation
# ----------------------------
def generate_answer(llm: ChatOpenAI, question: str, context: str) -> str:
    if not context.strip():
        return normalize_answer(
            "1) Overview\nNot enough information in the retrieved pages.\n"
            "2) Causes / Risk factors\nNot enough information in the retrieved pages.\n"
            "3) Symptoms\nNot enough information in the retrieved pages.\n"
            "4) Diagnosis\nNot enough information in the retrieved pages.\n"
            "5) Treatment / What you can do\nNot enough information in the retrieved pages.\n"
            "6) When to seek urgent care\nNot enough information in the retrieved pages."
        )

    msg = f"""You are a careful medical information assistant.

RULES:
- Use ONLY the provided MedlinePlus context.
- Do NOT guess or add outside knowledge.
- If a section is missing, write exactly: Not enough information in the retrieved pages.
- Do NOT include URLs in the answer body.
- NO blank lines between a heading and its content.

Use these headings EXACTLY:
1) Overview
2) Causes / Risk factors
3) Symptoms
4) Diagnosis
5) Treatment / What you can do
6) When to seek urgent care

CONTEXT:
{context}

QUESTION:
{question}
"""
    return normalize_answer(llm.invoke(msg).content)


# ----------------------------
# Main
# ----------------------------
def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if sys.stderr.encoding != "utf-8":
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="chroma_db", embedding_function=emb)

    question = extract_last_question(sys.stdin.read())
    if not question:
        print("ERROR: No question provided.", file=sys.stderr)
        sys.exit(1)

    # Agentic: section-based retrieval pool
    sq = section_queries(llm, question)

    pool: List[Tuple[Document, float]] = []
    for _, q in sq.items():
        pool.extend(retrieve(db, q, k=60))

    grouped = group_best_chunk_per_url(pool)
    picked = select_top_k(grouped, k=5)

    # If still not enough, do one broader hop
    if len(picked) < 5:
        expanded = f"{question} symptoms causes diagnosis treatment emergency"
        pool2 = pool + retrieve(db, expanded, k=120)
        grouped2 = group_best_chunk_per_url(pool2)
        picked = select_top_k(grouped2, k=5)

    ctx = build_context(picked, max_chars=8000, per_source=1400)
    ans = generate_answer(llm, question, ctx)

    print(ans)

    print("\nSources used (5 unique)")
    if not picked:
        print("No sources found.")
    else:
        for i, (_, s, u) in enumerate(picked[:5], 1):
            print(f"{i}. {u} (similarity: {s:.3f})")


if __name__ == "__main__":
    main()
