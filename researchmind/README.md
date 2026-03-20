# 🧠 ResearchMind — Agentic AI Research Assistant

> **An agentic AI workflow powered by Endee vector database, OpenAI function-calling, and Streamlit.**

ResearchMind lets you upload research documents, stores them as vector embeddings in **Endee**, and answers complex research questions using a multi-step AI agent that autonomously retrieves, reasons, and synthesizes answers.

---

## 📌 Problem Statement

Researchers and knowledge workers often have large collections of documents—papers, reports, notes—and need to extract insights quickly. Traditional keyword search is brittle. This project solves that by:

1. Embedding documents into Endee's high-performance vector database
2. Running an **agentic AI loop** that decides what to search, retrieves semantically relevant passages, and synthesizes a grounded answer
3. Showing full transparency via an **Agent Trace** panel that displays each reasoning step

---

## 🏗️ System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                             │
│  ┌──────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │  Doc Upload  │   │   Chat Panel    │   │  Agent Trace    │  │
│  │  + Ingest    │   │  (Q&A turns)    │   │  (step-by-step) │  │
│  └──────┬───────┘   └────────┬────────┘   └────────┬────────┘  │
└─────────┼───────────────────┼─────────────────────┼────────────┘
          │                   │                     │
          ▼                   ▼                     │
   ┌─────────────┐    ┌──────────────┐              │
   │  OpenAI     │    │  Research    │◄─────────────┘
   │  Embeddings │    │  Agent Loop  │
   │  (3-small)  │    │  (GPT-4o-mini│
   └──────┬──────┘    │  + Tools)    │
          │           └──────┬───────┘
          │                  │ tool calls
          ▼                  ▼
   ┌─────────────────────────────────┐
   │        Endee Vector DB          │
   │  (HNSW index, cosine, INT8)     │
   │  localhost:8080  via Python SDK │
   └─────────────────────────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| `app.py` | Streamlit UI — upload, chat, sidebar controls |
| `agent.py` | Agentic loop — GPT-4o-mini with 2 tools: `search_memory` + `synthesize` |
| `endee_client.py` | Endee SDK wrapper — index creation, upsert, vector search |
| `docker-compose.yml` | Spins up the Endee server in one command |

---

## 🔧 How Endee Is Used

Endee is used as the **long-term vector memory** of the agent:

### 1. Index Creation
```python
client.create_index(
    name="researchmind",
    dimension=1536,          # OpenAI text-embedding-3-small
    space_type="cosine",
    precision=Precision.INT8 # compressed storage for speed
)
```

### 2. Document Ingestion (Upsert)
Documents are chunked into ~400-word passages with 50-word overlap, embedded via OpenAI, and upserted:
```python
index.upsert([{
    "id": "paper_0",
    "vector": embedding,
    "meta": {"text": "...", "source": "paper.txt", "chunk": 0}
}])
```

### 3. Semantic Search (Agent Tool)
When the agent calls `search_memory`, Endee retrieves the nearest vectors:
```python
results = index.query(vector=query_embedding, top_k=5)
# Returns: id, similarity score, metadata (raw text + source)
```

The agent receives the retrieved text chunks and uses them to reason and compose a final answer.

---

## 🤖 Agent Workflow

The agent runs a **ReAct-style loop** using OpenAI function-calling:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Step 1: Model decides to call search_memory("query")   │
│  Step 2: Endee returns top-k relevant chunks            │
│  Step 3: Model reads chunks, may search again           │
│  Step 4: Model calls synthesize("final answer")         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
 Final Answer displayed in UI
```

Each step is logged and shown in the **Agent Trace** panel in real time.

---

## 🚀 Setup & Execution

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- OpenAI API key

### Step 1: Clone & Fork
```bash
# 1. Star the repo: https://github.com/endee-io/endee
# 2. Fork it to your GitHub account
# 3. Clone your fork
git clone https://github.com/<your-username>/endee
cd endee
```

### Step 2: Start Endee Server
```bash
docker compose up -d
# Verify: http://localhost:8080
```

### Step 3: Set Up Python Environment
```bash
# From the project root (or a new folder)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 5: Use the App
1. Enter your **OpenAI API key** in the sidebar
2. Click **Connect to Endee**
3. Upload `.txt` or `.md` files and click **Ingest to Endee**
4. Type a research question and click **Run Agent**
5. Watch the **Agent Trace** panel as the agent searches and reasons

---

## 📁 Project Structure

```
researchmind/
├── app.py               # Streamlit UI
├── agent.py             # Agentic loop (ReAct + function calling)
├── endee_client.py      # Endee Python SDK wrapper
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Endee server setup
└── README.md
```

---

## 🌟 Sample Use Cases

| Question | Agent Behavior |
|----------|---------------|
| "What are the key findings in my documents?" | Searches broadly, synthesizes a summary |
| "Compare the methodologies across papers" | Makes 2–3 targeted searches, then compares |
| "What evidence supports X hypothesis?" | Searches for X, retrieves supporting chunks |

---

## 🛠️ Tech Stack

- **Vector DB**: [Endee](https://github.com/endee-io/endee) (HNSW, cosine similarity, INT8 precision)
- **Embeddings**: OpenAI `text-embedding-3-small` (1536d)
- **LLM**: OpenAI `gpt-4o-mini` with function-calling
- **UI**: Streamlit
- **Language**: Python 3.11+

---

## 📜 License

This project is built on top of Endee, which is licensed under the Apache 2.0 License.
