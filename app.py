import subprocess
import re
import html
import os
import streamlit as st

st.set_page_config(page_title="Agentic RAG vs Vanilla RAG", layout="wide")

# ----------------------------
# Helpers: clean + parse
# ----------------------------
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ASK_LINE_RE = re.compile(r"^Ask a healthcare question:\s*", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s\)]+", re.IGNORECASE)


def clean_output(text: str) -> str:
    t = ANSI_RE.sub("", text or "")
    t = ASK_LINE_RE.sub("", t.lstrip())
    return t


def normalize_answer_spacing(text: str) -> str:
    """
    Reduce excessive blank lines / spacing in model outputs,
    and fix the '2.' then blank line then 'Causes...' pattern.
    """
    t = (text or "").replace("\r\n", "\n")
    t = "\n".join(line.rstrip() for line in t.splitlines())

    # collapse 3+ blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    # join:
    #   2.
    #
    #   Causes / Risk factors
    # into:
    #   2) Causes / Risk factors
    t = re.sub(
        r"(?m)^\s*(\d+)[\)\.]\s*\n\s*\n\s*([^\n]+)\s*$",
        r"\1) \2",
        t,
    )

    # remove extra blank lines right after headings like "1) Overview"
    t = re.sub(r"(?m)^(\d\)\s[^\n]+)\n\n+", r"\1\n", t)

    return t.strip()


def parse_sources(text: str):
    """Return unique base URLs (strip #fragment)."""
    urls = URL_RE.findall(text or "")
    seen, out = set(), []
    for u in urls:
        base = u.split("#")[0]
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def split_answer_and_sources(clean_text: str):
    """
    Remove sources section from Answer tab to avoid duplication.
    We split on common markers produced by your scripts.
    """
    markers = [
        "\nSources used",
        "\nSOURCE URLs used",
        "\nSources considered",
        "\nSOURCES",
        "\nSources used (unique",
        "\nNo sources found",
    ]
    idx = None
    for m in markers:
        pos = clean_text.find(m)
        if pos != -1:
            idx = pos
            break

    if idx is None:
        return clean_text.strip(), ""

    answer_part = clean_text[:idx].strip()
    sources_part = clean_text[idx:].strip()
    return answer_part, sources_part


def build_followup_prompt(user_question: str, history: list[dict], max_turns: int = 4) -> str:
    """
    Keep follow-up support: combine last turns into a single input.
    NOTE: Your rag scripts should read the last QUESTION: ... section.
    """
    user_question = (user_question or "").strip()

    policy = """INSTRUCTIONS (must follow):
- Answer using ONLY retrieved MedlinePlus content (no guessing).
- Use AT LEAST 5 UNIQUE MedlinePlus URLs in your retrieval process (different pages, not just #anchors).
- Write the answer with headings:
  1) Overview
  2) Causes / Risk factors
  3) Symptoms
  4) Diagnosis
  5) Treatment / What you can do
  6) When to seek urgent care
- Do NOT paste source links inside the answer body. Put sources in the Sources section only.
"""

    if not history:
        return f"{policy}\nQUESTION:\n{user_question}".strip()

    turns = history[-max_turns:]
    convo = []
    for t in turns:
        convo.append(f"User: {t['q']}")
        convo.append(f"Assistant: {t['a']}")
    convo_text = "\n".join(convo)

    return f"""{policy}

Conversation so far:
{convo_text}

QUESTION:
{user_question}
""".strip()


def render_plain_text_box(text: str):
    # Escape HTML so nothing becomes clickable/blue in Answer tab
    safe = html.escape(text or "")
    st.markdown(f'<div class="answer-box">{safe}</div>', unsafe_allow_html=True)


# ----------------------------
# Session state init
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Use TWO keys:
# - question_textarea is the widget key (Streamlit owns it)
# - current_question is our own copy
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if "last_vanilla" not in st.session_state:
    st.session_state.last_vanilla = ""
if "last_agentic" not in st.session_state:
    st.session_state.last_agentic = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False


# ----------------------------
# Styling (Blue theme + NO code highlighting)
# ----------------------------
st.markdown(
    """
<style>
.stApp { background: #0f172a; }
.block-container { padding-top: 2rem; }

h1, h2, h3, h4 { color: #60a5fa !important; }
label { color: #cbd5f5 !important; }

textarea, input {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-radius: 10px !important;
    border: 1px solid #2563eb !important;
}

div[data-baseweb="select"] > div {
    background-color: #020617 !important;
    color: white !important;
    border: 1px solid #2563eb !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(90deg,#2563eb,#3b82f6) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.55rem 1rem !important;
}

/* Plain text answer box (NO syntax highlighting / no blue links) */
.answer-box {
    background: #020617;
    color: #e5e7eb;
    border: 1px solid #2563eb;
    border-radius: 12px;
    padding: 16px;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    font-size: 0.95rem;
    line-height: 1.6;
    max-height: 600px;
    overflow-y: auto;
}

/* Hide Streamlit status widget (CONNECTING) */
[data-testid="stStatusWidget"] { display: none !important; }

/* Remove Streamlit header/footer */
header, footer { visibility: hidden; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
.stTabs [data-baseweb="tab"] {
    background-color: #1e293b;
    border-radius: 8px 8px 0 0;
    color: #94a3b8;
    padding: 8px 16px;
    border: 1px solid #334155;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background-color: #020617 !important;
    color: #60a5fa !important;
    border-color: #2563eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Header
# ----------------------------
st.title("🔵 Agentic RAG vs Vanilla RAG")
st.caption("MedlinePlus Knowledge Base • LangChain • Chroma • Streamlit")

# Sample questions (3)
st.markdown("### 💡 Try sample questions")
samples = [
    "What is high blood pressure and what are its common symptoms?",
    "What are the side effects of antibiotics?",
    "What should I do if I have bipolar disorder and insomnia?",
]
cols = st.columns(3)
for i, q in enumerate(samples):
    if cols[i].button(q, key=f"sample_{i}", use_container_width=True):
        # ✅ Set BEFORE widget is created (safe)
        st.session_state.current_question = q
        st.session_state.show_results = False
        st.rerun()

st.markdown("---")

mode = st.selectbox("🔧 Run mode", ["Vanilla", "Agentic", "Both"], index=2)

# Input
question_text = st.text_area(
    "📝 Enter your question",
    value=st.session_state.current_question,
    height=90,
    placeholder="Type your healthcare question here...",
    key="question_textarea",
)
st.session_state.current_question = question_text

# Buttons (Clear nearer to Run)
b1, b2, _sp = st.columns([1, 1, 8])
with b1:
    run_btn = st.button("▶️ Run", type="primary", use_container_width=True)
with b2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

# Clear: just clear the input + hide results (no errors)
if clear_btn:
    st.session_state.current_question = ""
    st.session_state.show_results = False
    st.session_state.last_vanilla = ""
    st.session_state.last_agentic = ""
    st.rerun()


# ----------------------------
# Runner
# ----------------------------
def run_script(script_path: str, q: str) -> str:
    """
    Run a Python script with the question via stdin.
    Return stdout OR a friendly error message.
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    p = subprocess.run(
        ["uv", "run", "python", script_path],
        input=(q + "\n"),
        text=True,
        capture_output=True,
        shell=False,
        timeout=120,
        encoding="utf-8",
        errors="replace",
        env=env,
    )

    if p.returncode != 0:
        err = clean_output(p.stderr)
        out = clean_output(p.stdout)
        return f"⚠️ Error running {script_path}\n\n{err}\n\n{out}".strip()

    return p.stdout


def render_result(title: str, raw_out: str):
    cleaned = clean_output(raw_out)
    answer_only, sources_block = split_answer_and_sources(cleaned)

    answer_only = normalize_answer_spacing(answer_only)

    urls = parse_sources(cleaned)
    unique_medline = [u for u in urls if "medlineplus.gov" in u.lower()]

    st.subheader(title)

    tab1, tab2 = st.tabs(["📄 Answer", "🔗 Sources"])

    with tab1:
        if answer_only.strip():
            render_plain_text_box(answer_only)
        else:
            st.info("No answer content found.")

    with tab2:
        if len(unique_medline) < 5:
            st.warning(f"Only found **{len(unique_medline)}** unique MedlinePlus URLs (target: ≥ 5).")
        else:
            st.success(f"✅ Found **{len(unique_medline)}** unique MedlinePlus URLs")

        if not unique_medline:
            st.info("No MedlinePlus sources found in output.")
        else:
            st.markdown("#### Referenced Sources")
            for idx, u in enumerate(unique_medline[:10], 1):
                st.markdown(f"{idx}. [{u}]({u})")

        if sources_block:
            with st.expander("View raw sources section"):
                render_plain_text_box(sources_block)


# ----------------------------
# Run logic
# ----------------------------
if run_btn:
    user_q = (st.session_state.current_question or "").strip()
    if not user_q:
        st.warning("⚠️ Please enter a question before running.")
        st.stop()

    combined_question = build_followup_prompt(user_q, st.session_state.history, max_turns=4)
    st.session_state.show_results = True

    with st.spinner("🔄 Processing..."):
        if mode == "Vanilla":
            out = run_script("src/vanilla_rag.py", combined_question)
            st.session_state.last_vanilla = out
            st.session_state.history.append({"q": user_q, "a": clean_output(out)})

        elif mode == "Agentic":
            out = run_script("src/agentic_rag.py", combined_question)
            st.session_state.last_agentic = out
            st.session_state.history.append({"q": user_q, "a": clean_output(out)})

        else:
            out_v = run_script("src/vanilla_rag.py", combined_question)
            out_a = run_script("src/agentic_rag.py", combined_question)
            st.session_state.last_vanilla = out_v
            st.session_state.last_agentic = out_a
            merged = f"[Vanilla]\n{clean_output(out_v)}\n\n[Agentic]\n{clean_output(out_a)}"
            st.session_state.history.append({"q": user_q, "a": merged})


# ----------------------------
# Display results
# ----------------------------
if st.session_state.show_results:
    st.markdown("---")
    st.markdown("## 📊 Results")

    if mode == "Vanilla" and st.session_state.last_vanilla:
        render_result("🔹 Vanilla RAG", st.session_state.last_vanilla)
    elif mode == "Agentic" and st.session_state.last_agentic:
        render_result("🔸 Agentic RAG", st.session_state.last_agentic)
    elif mode == "Both":
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.last_vanilla:
                render_result("🔹 Vanilla RAG", st.session_state.last_vanilla)
        with col2:
            if st.session_state.last_agentic:
                render_result("🔸 Agentic RAG", st.session_state.last_agentic)


# ----------------------------
# Conversation history
# ----------------------------
if st.session_state.history:
    st.markdown("---")
    with st.expander(f"📜 Conversation History ({len(st.session_state.history)} turns)"):
        for i, turn in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(f"**Q{i}:** {turn['q']}")
            preview = normalize_answer_spacing(turn["a"])[:250]
            st.caption(f"Answer preview: {preview}...")
            st.markdown("---")
