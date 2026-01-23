# LangChain Advanced RAG

A modular Python project for Retrieval-Augmented Generation (RAG) with LCEL chains, supporting PostgreSQL/PGVector and Pinecone as vector stores.

## Features

- **Multiple RAG Strategies**: Base, Query Rewriter, Multi-Query, HyDE, and Full Context chains
- **Dual Backend Support**: PostgreSQL/PGVector and Pinecone
- **Adaptive Chunking**: Automatically adjusts chunk size and tokenizer based on the selected embedding model (e.g., Gemini's 2k window vs. BGE's 512)
- **Multiple LLM Providers**: Gemini, Groq, Ollama, Perplexity
- **RAGAS Evaluation**: Faithfulness, Answer Correctness, Context Precision/Recall
- **Synthetic Q/A Generation**: Generate test cases from document chunks
- **Deduplication**: Content hashing with deterministic IDs

## Project Structure

```
langchain-advanced-rag/
├── src/app/
│   ├── config.py           # Settings, LLM/Embeddings factories
│   ├── logging_conf.py     # Logging configuration
│   ├── utils/
│   │   ├── hashing.py      # Content hashing
│   │   ├── retry.py        # Batch retry with backoff
│   │   └── dedup.py        # Document deduplication
│   ├── vectorstores/
│   │   ├── postgres_pgvector.py
│   │   └── pinecone_store.py
│   ├── rag/
│   │   ├── prompts.py      # All prompt templates
│   │   ├── lcel_helpers.py # LCEL utility functions
│   │   └── chains.py       # RAG chain builders
│   └── eval/
│       ├── ragas_eval.py   # RAGAS evaluation
│       └── synthetic_qa.py # Synthetic Q/A generation
├── scripts/
│   ├── ask.py              # Ask questions via CLI
│   ├── evaluate_ragas.py   # Run RAGAS evaluation
│   ├── generate_synthetic.py
│   ├── ingest_postgres.py
│   ├── ingest_pinecone.py
│   ├── bootstrap_postgres.py
│   └── bootstrap_pinecone.py
├── documents/              # Your PDF documents
├── requirements.txt
└── .env
```

## Setup

### 1. Clone and Install

```bash
git clone <repo>
cd langchain-advanced-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Required variables:
```env
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
POSTGRES_URL=postgresql+psycopg://user:pass@localhost:5432/dbname

# For Pinecone (optional)
PINECONE_API_KEY=your_key
PINECONE_INDEX=your_index
PINECONE_DIMENSION=3072
```

### 3. Start PostgreSQL (Docker)

```bash
docker-compose up -d
```

### 4. Bootstrap Infrastructure

```bash
# Create table and index
python scripts/bootstrap_postgres.py --table documents_embeddings_gemini --vector-size 3072

# Or for Pinecone
python scripts/bootstrap_pinecone.py
```

### 5. Ingest Documents

Place your PDFs in the `documents/` folder, then:

```bash
python scripts/ingest_postgres.py

# Or for Pinecone
python scripts/ingest_pinecone.py
```

## Usage

### Ask Questions

```bash
# Basic usage
python scripts/ask.py "Como fazer um seguro viagem?"

# With options
python scripts/ask.py "Qual o limite do cartão?" --chain-type multiquery --show-contexts

# Chain types: base, rewriter, multiquery, hyde, full
python scripts/ask.py "..." --chain-type hyde --backend pinecone
```

### RAGAS Evaluation

```bash
# Single question
python scripts/evaluate_ragas.py \
  --question "Qual o limite máximo de benefício?" \
  --ground-truth "Até USD 200 por incidente"

# Batch evaluation from JSON file
python scripts/evaluate_ragas.py --input-file test_cases.json --output-prefix my_eval
```

### Generate Synthetic Q/A

```bash
# Generate Q/A pairs
python scripts/generate_synthetic.py --sample-size 20

# Generate and evaluate
python scripts/generate_synthetic.py --sample-size 10 --evaluate
```

## RAG Chain Types

| Chain | Description |
|-------|-------------|
| `base` | Simple retriever → LLM |
| `rewriter` | Rewrite query for better retrieval |
| `multiquery` | Generate 5 versions, batch retrieve, dedupe |
| `hyde` | Generate hypothetical answer, use for retrieval |
| `full` | Returns `{query, answer, contexts, docs}` |

## Architecture

```mermaid
graph LR
    A[Query] --> B{Chain Type}
    B -->|base| C[Retriever]
    B -->|rewriter| D[Rewrite] --> C
    B -->|multiquery| E[Generate 5 queries] --> F[Batch Retrieve] --> G[Dedupe]
    B -->|hyde| H[Generate Hypothetical] --> C
    C --> I[Format Context]
    G --> I
    I --> J[LLM]
    J --> K[Answer]
```

## License

MIT
