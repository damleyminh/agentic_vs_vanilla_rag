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
1️⃣ Create `.env`

Create a file named `.env` in the project root:

```env
OPENAI_API_KEY=your_openai_key_here


2️⃣ Install dependencies (using uv)
uv sync


If you don’t have uv:

pip install uv

---
## How to Run
✅ Step 1 — Build the vector database (run ONCE)
This scrapes MedlinePlus and creates chroma_db.
uv run python src/ingest.py

⚠️ You MUST run this before Streamlit.
Otherwise RAG will return empty answers.

✅ Step 2 — Launch Streamlit app
uv run streamlit run app.py
Then open:
http://localhost:8501

💡 Example Questions
What are the side effects of antibiotics?
What is high blood pressure?
What should I do if I have bipolar disorder and insomnia?
What causes type 2 diabetes?
When should I seek urgent care for chest pain?

⚠️ Notes / Safety

This project is for educational purposes only.
It does NOT provide medical advice.

Always consult a healthcare professional.




