# src/ingest.py
from __future__ import annotations

import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from scrape import crawl_site
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    load_dotenv()

    # Base domain (keep from your .env)
    base_url = os.environ.get("BASE_URL", "https://medlineplus.gov/").strip()

    # Force-ingest important topic pages so RAG can answer real questions
    extra_seeds = [
        "https://medlineplus.gov/diabetes.html",
        "https://medlineplus.gov/diabetestype2.html",
        "https://medlineplus.gov/diabetestype1.html",
        "https://medlineplus.gov/highbloodpressure.html",
        "https://medlineplus.gov/bloodpressure.html",
        "https://medlineplus.gov/antibiotics.html",
        "https://medlineplus.gov/bipolardisorder.html",
        "https://medlineplus.gov/insomnia.html",
        "https://medlineplus.gov/heartdiseases.html",
        "https://medlineplus.gov/cholesterol.html",
        "https://medlineplus.gov/asthma.html",
        "https://medlineplus.gov/arthritis.html",
        "https://medlineplus.gov/depression.html",
        "https://medlineplus.gov/anxiety.html",
        "https://medlineplus.gov/cancer.html",
        "https://medlineplus.gov/stroke.html",
        "https://medlineplus.gov/obesity.html",
        "https://medlineplus.gov/heartattack.html",
        "https://medlineplus.gov/flu.html",
        "https://medlineplus.gov/commoncold.html",
    ]

    # Crawl with extra seeds (your updated scrape.py supports extra_seeds)
    pages = crawl_site(base_url, max_pages=40, extra_seeds=extra_seeds)

    docs = [
        Document(page_content=p["text"], metadata={"source": p["url"]})
        for p in pages
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Rebuild the DB in ./chroma_db
    db = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory="chroma_db"
    )
    db.persist()

    print(f"Ingested pages={len(pages)}, chunks={len(chunks)} into ./chroma_db")
    print("Sample sources:")
    for u in [p["url"] for p in pages[:10]]:
        print(" -", u)


if __name__ == "__main__":
    main()