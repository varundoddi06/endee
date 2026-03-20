import streamlit as st
import json
import time
from agent import ResearchAgent
from endee_client import EndeeVectorStore

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
}

.stApp {
    background: #0a0a0f;
    color: #e8e4d9;
}

.main-header {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f5c842 0%, #ff7043 50%, #e040fb 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.sub-header {
    color: #6b6b7a;
    font-size: 0.85rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 0;
}

.agent-step {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #f5c842;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.82rem;
    color: #a8a8b8;
    font-family: 'DM Mono', monospace;
}

.agent-step.tool {
    border-left-color: #ff7043;
}

.agent-step.memory {
    border-left-color: #40c8e0;
}

.agent-step.answer {
    border-left-color: #7cfc7c;
    color: #e8e4d9;
    font-size: 0.9rem;
}

.metric-box {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}

.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f5c842;
}

.metric-label {
    font-size: 0.7rem;
    color: #6b6b7a;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.doc-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.8rem;
    color: #a8a8b8;
}

.doc-card strong {
    color: #e8e4d9;
    font-family: 'Syne', sans-serif;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #12121a !important;
    border: 1px solid #1e1e2e !important;
    color: #e8e4d9 !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #f5c842, #ff7043) !important;
    color: #0a0a0f !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    padding: 0.5rem 2rem !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px);
}

.sidebar .stButton > button {
    background: #1e1e2e !important;
    color: #e8e4d9 !important;
    width: 100%;
}

div[data-testid="stSidebar"] {
    background: #0d0d14 !important;
    border-right: 1px solid #1e1e2e;
}

.chat-user {
    background: #1a1a28;
    border-radius: 10px 10px 2px 10px;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #e8e4d9;
    text-align: right;
    border: 1px solid #2a2a3e;
}

.chat-agent {
    background: #12121a;
    border-radius: 2px 10px 10px 10px;
    padding: 10px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    color: #e8e4d9;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #f5c842;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-dot.green { background: #7cfc7c; }
.status-dot.red { background: #ff5555; }
.status-dot.yellow { background: #f5c842; }
</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []
if "docs_ingested" not in st.session_state:
    st.session_state.docs_ingested = []
if "endee_connected" not in st.session_state:
    st.session_state.endee_connected = False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-header" style="font-size:1.6rem;">🧠 ResearchMind</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Agentic AI + Endee</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### ⚙️ Endee Connection")
    endee_host = st.text_input("Host", value="http://localhost:8080", key="endee_host")
    endee_token = st.text_input("Auth Token (optional)", type="password", key="endee_token")
    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key",
                               help="Used for embeddings and LLM calls")

    if st.button("🔌 Connect to Endee"):
        try:
            vs = EndeeVectorStore(base_url=endee_host, auth_token=endee_token or None)
            vs.ensure_index()
            st.session_state.endee_connected = True
            st.session_state.vector_store = vs
            st.success("Connected!")
        except Exception as e:
            st.session_state.endee_connected = False
            st.error(f"Failed: {e}")

    status = "green" if st.session_state.endee_connected else "red"
    label = "Connected" if st.session_state.endee_connected else "Not connected"
    st.markdown(f'<span class="status-dot {status}"></span><span style="font-size:0.8rem;color:#6b6b7a;">{label}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📄 Ingest Documents")
    uploaded = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], accept_multiple_files=True)

    if st.button("📥 Ingest to Endee") and uploaded:
        if not st.session_state.endee_connected:
            st.warning("Connect to Endee first!")
        elif not openai_key:
            st.warning("Enter OpenAI API key first!")
        else:
            with st.spinner("Ingesting..."):
                vs = st.session_state.vector_store
                for f in uploaded:
                    text = f.read().decode("utf-8")
                    chunks = chunk_text(text, chunk_size=400)
                    for i, chunk in enumerate(chunks):
                        vs.upsert_chunk(
                            doc_id=f"{f.name}_{i}",
                            text=chunk,
                            meta={"source": f.name, "chunk": i},
                            openai_key=openai_key
                        )
                    if f.name not in st.session_state.docs_ingested:
                        st.session_state.docs_ingested.append(f.name)
            st.success(f"Ingested {len(uploaded)} file(s)!")

    if st.session_state.docs_ingested:
        st.markdown("**Ingested:**")
        for doc in st.session_state.docs_ingested:
            st.markdown(f'<div class="doc-card"><strong>📄</strong> {doc}</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.agent_logs = []
        st.rerun()

# ── Main Area ────────────────────────────────────────────────────────────────
col_title, col_metrics = st.columns([3, 2])

with col_title:
    st.markdown('<h1 class="main-header">ResearchMind</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Agentic AI Research Assistant · Powered by Endee Vector DB</p>', unsafe_allow_html=True)

with col_metrics:
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'''<div class="metric-box">
            <div class="metric-val">{len(st.session_state.docs_ingested)}</div>
            <div class="metric-label">Docs</div></div>''', unsafe_allow_html=True)
    with m2:
        st.markdown(f'''<div class="metric-box">
            <div class="metric-val">{len(st.session_state.chat_history)}</div>
            <div class="metric-label">Turns</div></div>''', unsafe_allow_html=True)
    with m3:
        st.markdown(f'''<div class="metric-box">
            <div class="metric-val">{len(st.session_state.agent_logs)}</div>
            <div class="metric-label">Steps</div></div>''', unsafe_allow_html=True)

st.markdown("---")

# ── Chat + Trace Layout ───────────────────────────────────────────────────────
chat_col, trace_col = st.columns([3, 2])

with chat_col:
    st.markdown("### 💬 Research Chat")
    chat_container = st.container(height=480)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-agent">🧠 {msg["content"]}</div>', unsafe_allow_html=True)

    query = st.text_input("Ask a research question...", placeholder="e.g. Summarize the key findings about climate change from my documents", key="user_query")

    if st.button("🚀 Run Agent") and query:
        if not st.session_state.endee_connected:
            st.warning("Please connect to Endee first using the sidebar.")
        elif not openai_key:
            st.warning("Please enter your OpenAI API key in the sidebar.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.agent_logs = []

            agent = ResearchAgent(
                vector_store=st.session_state.vector_store,
                openai_key=openai_key
            )

            with st.spinner("Agent thinking..."):
                result = agent.run(query, log_callback=lambda log: st.session_state.agent_logs.append(log))

            st.session_state.chat_history.append({"role": "assistant", "content": result})
            st.rerun()

with trace_col:
    st.markdown("### 🔍 Agent Trace")
    trace_container = st.container(height=560)
    with trace_container:
        if not st.session_state.agent_logs:
            st.markdown('<p style="color:#3a3a4a;font-size:0.82rem;">Agent steps will appear here during execution...</p>', unsafe_allow_html=True)
        for log in st.session_state.agent_logs:
            css_class = log.get("type", "")
            icon = {"thought": "💭", "tool": "🔧", "memory": "🗃️", "answer": "✅"}.get(css_class, "→")
            st.markdown(f'<div class="agent-step {css_class}">{icon} <strong>{log.get("label","")}</strong><br>{log.get("content","")}</div>', unsafe_allow_html=True)


# ── Helper ────────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400) -> list[str]:
    words = text.split()
    chunks, current = [], []
    for word in words:
        current.append(word)
        if len(current) >= chunk_size:
            chunks.append(" ".join(current))
            current = current[-50:]  # 50-word overlap
    if current:
        chunks.append(" ".join(current))
    return chunks
