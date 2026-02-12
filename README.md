# Agentic RAG vs Vanilla RAG (MedlinePlus)

A Streamlit project that compares **Vanilla RAG** vs **Agentic RAG** for medical Q&A using a **MedlinePlus** knowledge base.  
The system ingests MedlinePlus pages, stores chunks in **ChromaDB**, and answers questions using **retrieved context only**.

---

## Features

- ✅ MedlinePlus web crawling + ingestion
- ✅ Vector database with Chroma (`chroma_db/`)
- ✅ **Vanilla RAG**: single-shot retrieval + answer generation
- ✅ **Agentic RAG**: section-based retrieval (overview/causes/symptoms/diagnosis/treatment/urgent care)
- ✅ Streamlit UI with:
  - Run Vanilla / Agentic / Both
  - Answer + Sources tabs
  - Clear button resets only the input box


---

## Requirements

- Python 3.10+ (recommended)
- `uv` package manager
- OpenAI API key (stored in `.env`)

---

## Setup

### 1) Create `.env`
Create a file named `.env` in the project root:

```bash
OPENAI_API_KEY=your_key_here
BASE_URL=https://medlineplus.gov/

---

## How It Works
# Vanilla RAG

 - Retrieves relevant chunks from Chroma using the question
 - Builds a context from top unique MedlinePlus sources
 - Generates a structured answer using ONLY retrieved context

# Agentic RAG

 - Generates sub-queries for each section:
 - Overview, Causes, Symptoms, Diagnosis, Treatment, Urgent Care
 - Retrieves documents for each sub-query
 - Merges results and picks the best unique sources
 - Generates a structured answer using ONLY retrieved context

# Notes / Safety
This project provides information only and is not medical advice.

If retrieved pages do not contain enough information, the model should say:
"Not enough information in the retrieved pages."




