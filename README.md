<div align="center">

# 🚀 Advanced Agentic RAG Platform

### Production-Grade Parent-Child Retrieval with Semantic Chunking & Cross-Encoder Reranking

![Status](https://img.shields.io/badge/Status-Production--Ready-22c55e)
![Architecture](https://img.shields.io/badge/Architecture-Parent--Child%20Nodes-orange)
![LLM](https://img.shields.io/badge/LLM-Groq%20(Llama%203)-f59e0b)
![VectorDB](https://img.shields.io/badge/Vector-Qdrant-red)

</div>

---

## ✨ System Evolution
This project has evolved from a basic PDF-to-Chat tool into a **High-Precision RAG Pipeline**. By decoupling retrieval (Child Nodes) from context (Parent Nodes) and adding a reranking layer, the system achieves significantly higher groundedness and accuracy than standard RAG implementations.

---

## 🏗️ Technical Architecture (The 2-Stage Pipeline)

### 1. Ingestion Layer (Hybrid Storage)
- **Parent Nodes:** The original document is parsed into large, logical sections (Parent Nodes) and stored in **PostgreSQL**. This preserves full context for the LLM.
- **Child Nodes:** Each Parent Node is split into smaller, **Semantic Chunks** (Child Nodes).
- **Vector Index:** Child Nodes are embedded and stored in **Qdrant** with a `parent_id` reference in the metadata.

### 2. Retrieval & Generation Layer
- **Semantic Retrieval:** The user query is embedded to find the top-K relevant **Child Nodes** from Qdrant.
- **Cross-Encoder Reranking:** Retrieved chunks are re-scored against the query to ensure only the most relevant hits survive.
- **Context Re-construction:** The system uses the `parent_id` of the top reranked chunks to fetch the **Full Parent Context** from PostgreSQL.
- **Inference:** The query + Full Parent Context is sent to **Groq (Llama 3)** for lightning-fast, highly accurate generation.

---

## 🔄 Workflow Diagrams

### End-to-End Data Flow

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI Orchestrator
    participant PG as PostgreSQL (Parent Store)
    participant Q as Qdrant (Child Vector Store)
    participant R as Reranker (Cross-Encoder)
    participant LLM as Groq (Llama 3)

    User->>API: Upload Document
    API->>PG: Store Parent Nodes
    API->>Q: Store Child Node Embeddings + parent_id

    User->>API: User Query
    API->>Q: Search top-20 Child Nodes
    Q-->>API: Returns Candidates
    API->>R: Rerank Candidates
    R-->>API: Returns top-5 relevant Chunks
    API->>PG: Fetch Parent Nodes (using parent_id)
    PG-->>API: Full Context Data
    API->>LLM: Full Context + Query + Prompt
    LLM-->>User: Grounded, Context-Aware Response
