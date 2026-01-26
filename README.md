![Python](https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green?style=for-the-badge&logo=chainlink&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red?style=for-the-badge&logo=streamlit&logoColor=white)
![Supabase](https://img.shields.io/badge/Supabase-PGVector-emerald?style=for-the-badge&logo=supabase&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-purple?style=for-the-badge)

# 🧠 LangChain Advanced RAG

> **Production-ready Retrieval-Augmented Generation system featuring Adaptive Chunking, Agentic RAG Chains, and RAGAS Evaluation.**

This project implements a modular, high-performance RAG pipeline designed to solve common production challenges like hallucination, poor recall, and lack of observability. It supports both **PostgreSQL (Supabase/PGVector)** and **Pinecone** as vector backends.

<div align="center">
  <img src="images/chat-demo-02.png" alt="Chat Demo" width="45%">
  <img src="images/ragas-dashboard-01.png" alt="Ragas Dashboard" width="45%">
</div>

---

## 📚 Table of Contents

- [🧠 LangChain Advanced RAG](#-langchain-advanced-rag)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Features](#-features)
    - [Core RAG Capabilities](#core-rag-capabilities)
    - [Advanced RAG Chains](#advanced-rag-chains)
  - [🏗 Architecture](#-architecture)
  - [🛠 Tech Stack](#-tech-stack)
  - [📂 Project Structure](#-project-structure)
  - [⚡ Getting Started](#-getting-started)
    - [1. Clone \& Env](#1-clone--env)
    - [2. Configure Credentials](#2-configure-credentials)
    - [3. Bootstrap Database](#3-bootstrap-database)
    - [4. Ingest Documents](#4-ingest-documents)
  - [🖥 Usage](#-usage)
    - [Streamlit UI](#streamlit-ui)
    - [CLI Tools](#cli-tools)
  - [📊 Evaluation](#-evaluation)
  - [📄 License](#-license)

---

## 🚀 Features

### Core RAG Capabilities
- **Adaptive Chunking**: Dynamically adjusts chunk sizes based on the embedding model's context window (e.g., Gemini vs BGE).
- **Dual Vector Backends**: Seamless switching between Supabase PGVector and Pinecone.
- **Deduplication**: Content hashing (`SHA-256`) to prevent duplicate document ingestion.
- **Robust Ingestion**: `PyPDFDirectoryLoader` with error handling for complex PDFs.

### Advanced RAG Chains
| Chain | Description | Use Case |
|-------|-------------|----------|
| **Base** | Standard `Retriever -> LLM` | Simple factual queries |
| **Rewriter** | LLM rewrites user query before retrieval | Ambiguous or poorly phrased queries |
| **Multi-Query** | Generates 5 variants of the query, retrieves for all | Complex queries requiring broad context |
| **HyDE** | Hypothetical Document Embeddings | Abstract or thematic queries |
| **Rerank** | Retrieves `Top-K` then uses LLM Judge to score relevance | High-precision requirements |

---

## 🏗 Architecture

```mermaid
graph LR
    User[User Query] --> Router{Chain Selection}
    
    subgraph "Retrieval Strategies"
        Router -->|Base| Ret[Retriever]
        Router -->|Rewriter| RW[Query Rewriter] --> Ret
        Router -->|MultiQuery| MQ[Generate 5 Queries] --> Batch[Batch Retrieve]
        Router -->|HyDE| HY[Generate Hypoth. Doc] --> Ret
        Router -->|Rerank| RR[Retrieve K=20] --> Judge[LLM Reranker] --> TopK[Top K=3]
    end

    Ret --> Context
    Batch --> Dedup[Deduplicate] --> Context
    TopK --> Context
    
    Context --> Augment[Context + Prompt]
    Augment --> LLM[Generation]
    LLM --> Answer
```

---

## 🛠 Tech Stack

- **Framework**: LangChain, LangGraph
- **LLMs**: Google Gemini (Flash/Pro), Groq (Llama 3, Mixtral), Perplexity, Ollama
- **Vector Stores**: Supabase (pgvector), Pinecone
- **Interface**: Streamlit (Chat + Dashboard)
- **Evaluation**: Ragas (Faithfulness, Correctness, Precision, Recall)
- **Observability**: Custom Logging, LangSmith (optional)

---

## 📂 Project Structure

```text
langchain-advanced-rag/
├── .streamlit/             # Streamlit Cloud configuration
├── src/
│   └── app/
│       ├── config.py       # Centralized configuration & factories
│       ├── vectorstores/   # PGVector & Pinecone connectors
│       ├── rag/            # RAG Chains & Prompt Templates
│       ├── eval/           # RAGAS metrics & Synthetic Data
│       └── utils/          # Hashing, Chunking, Retry logic
├── streamlit_app/          # UI Application
│   ├── app.py              # Main Chat Interface
│   ├── shared/             # Shared UI components
│   └── pages/              # Evaluation Dashboard
├── scripts/                # CLI Operational Scripts
│   ├── ingest_*.py         # Document Ingestion
│   ├── bootstrap_*.py      # Database Setup
│   └── evaluate_ragas.py   # Evaluation Runner
└── documents/              # PDF Source Directory
```

---

## ⚡ Getting Started

### 1. Clone & Env
```bash
git clone https://github.com/yourusername/langchain-advanced-rag.git
cd langchain-advanced-rag

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and populate your keys:
```ini
GOOGLE_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...
POSTGRES_URL=postgresql+psycopg://postgres:password@db.supabase.co:5432/postgres
PINECONE_API_KEY=pcsk_...
```

### 3. Bootstrap Database
Initialize the vector tables in your chosen backend:
```bash
# For Supabase/PostgreSQL
python scripts/bootstrap_postgres.py --table documents_embeddings_gemini

# For Pinecone
python scripts/bootstrap_pinecone.py
```

### 4. Ingest Documents
Place PDFs in `documents/` and run:
```bash
python scripts/ingest_postgres.py
# or
python scripts/ingest_pinecone.py
```

---

## 🖥 Usage

### Streamlit UI
Run the full web interface with Chat and Dashboard:
```bash
streamlit run streamlit_app/app.py
```
- **Chat**: Experiment with different chains (`rerank`, `multiquery`, etc.)
- **Dashboard**: Visualize RAGAS metrics via the sidebar page.

### CLI Tools
Quickly test via terminal:

```bash
# Ask a question
python scripts/ask.py "What is the coverage limit?" --chain-type rerank

# Run Evaluation
python scripts/evaluate_ragas.py --input-file synthetic_qa.json
```

---

## 📊 Evaluation

We use **RAGAS** to quantitatively measure pipeline performance.

1. **Generate Synthetic Data**:
   ```bash
   python scripts/generate_synthetic.py --sample-size 10
   ```
2. **Run Evaluation**:
   ```bash
   python scripts/evaluate_ragas.py --input-file synthetic_qa.json --output-prefix my_eval
   ```
3. **Analyze Results**:
   Open the **Evaluation Dashboard** in the Streamlit app to view radar charts and heatmaps.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
