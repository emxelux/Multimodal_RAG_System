---
title: ChatPDF
emoji: 🐳
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---


<<<<<<< HEAD
<div align="center">

# 🚀 AI Knowledge Platform — RAG Course

### Retrieval-Augmented Generation (RAG) System with Document Upload, Hybrid Search, and Conversational AI

![Status](https://img.shields.io/badge/Status-Architecture%20Ready-22c55e)
![Stack](https://img.shields.io/badge/Stack-HTML|CSS|JavaScript%20%7C%20FastAPI%20%7C%20PostgreSQL%20%7C%20Qdrant%20%7C%20LangChain-2563eb)
![Focus](https://img.shields.io/badge/Focus-Production%20Grade%20RAG-a855f7)

</div>

---

## ✨ Executive Summary

This repository presents a complete blueprint for an **AI Knowledge Platform** built on **Retrieval-Augmented Generation (RAG)** principles.
It combines:

- A user-friendly frontend for document upload and chat.
- A backend API for orchestration.
- A retrieval + generation AI pipeline.
- A structured PostgreSQL schema for product data.
- A vector search layer (Qdrant) for semantic retrieval.

The goal is to deliver answers that are **context-aware, traceable, and scalable** for real-world knowledge workflows.

---

## 🧭 Table of Contents

- [✨ Executive Summary](#-executive-summary)
- [🏗️ System Architecture](#️-system-architecture)
- [🖼️ Wireframes (UI Blueprint)](#️-wireframes-ui-blueprint)
- [🔄 Workflow Diagrams](#-workflow-diagrams)
- [🧱 Data Architecture](#-data-architecture)
- [🛠️ Implementation Guide](#️-implementation-guide)
- [🔐 Security](#-security)
- [📈 Scalability Notes](#-scalability-notes)
- [✅ Why This Project Stands Out](#-why-this-project-stands-out)

---

## 🏗️ System Architecture

### Core Components

```mermaid
flowchart LR
    U[User] --> F[Frontend\nWeb/Mobile UI]
    F --> A[API Layer]
    A --> D[Document Processing Service]
    A --> M[AI Model Service\nRAG Orchestrator]
    D --> P[(PostgreSQL)]
    D --> Q[(Qdrant Vector DB)]
    M --> P
    M --> Q
    M --> F
```

### Functional Breakdown

- **Frontend**
  - User interface for uploads and chat.
  - Real-time conversational experience.
- **API Layer**
  - Entry point for frontend requests.
  - Routes calls to processing and AI services.
- **Document Processing Service**
  - Extracts and normalizes text.
  - Splits content into chunks and stores metadata.
- **AI Model Service (RAG)**
  - Retrieves relevant chunks.
  - Generates grounded responses.
- **Storage Layer**
  - PostgreSQL for transactional and relational data.
  - Qdrant for vector embeddings and similarity search.

---

## 🖼️ Wireframes (UI Blueprint)

> Simple wireframe-style layouts to communicate product direction quickly.

### 1) Landing + Workspace

```mermaid
flowchart TB
    subgraph App Shell
      A[Top Nav\nLogo | Workspaces | Profile]
      B[Sidebar\n- Uploads\n- Documents\n- Conversations\n- Settings]
      C[Main Content\nWorkspace Overview\nRecent Docs + Recent Chats]
    end
    A --> B
    A --> C
```

### 2) Document Upload Screen

```mermaid
flowchart LR
    L[Left Pane\nDrag & Drop\nSupported Files\nUpload Progress] --> R[Right Pane\nExtracted Metadata\nChunk Preview\nIngestion Status]
```

### 3) Chat Experience Screen

```mermaid
flowchart TB
    H[Header\nConversation Title + Workspace]
    M[Message Thread\nUser + Assistant Bubbles\nSource Snippets]
    I[Input Area\nPrompt Box + Send + Attach]
    H --> M --> I
```

---

## 🔄 Workflow Diagrams

### End-to-End Data Flow

```mermaid
sequenceDiagram
    participant User
    participant UI as Frontend UI
    participant API as API Layer
    participant DP as Doc Processing
    participant DB as PostgreSQL
    participant VDB as Qdrant
    participant RAG as AI Model Service

    User->>UI: Upload document
    UI->>API: POST /documents
    API->>DP: Process file
    DP->>DB: Save document + metadata
    DP->>VDB: Save chunk embeddings

    User->>UI: Ask question in chat
    UI->>API: POST /chat
    API->>RAG: Forward user query
    RAG->>VDB: Retrieve relevant chunks
    RAG->>DB: Fetch metadata/history
    RAG-->>API: Generated grounded answer
    API-->>UI: Response + context
    UI-->>User: Render answer
```

### Retrieval + Generation Logic

```mermaid
flowchart TD
    Q1[User Query] --> E1[Embed Query]
    E1 --> R1[Vector Similarity Search]
    R1 --> C1[Top-K Context Chunks]
    C1 --> G1[LLM Prompt Assembly]
    G1 --> A1[Grounded Answer]
    A1 --> V1[Return Answer + Sources]
```

---

## 🧱 Data Architecture

### PostgreSQL Entity Relationship Diagram

```mermaid
erDiagram
    USERS ||--o{ WORKSPACES : owns
    WORKSPACES ||--o{ DOCUMENTS : contains
    DOCUMENTS ||--o{ CHUNKS : splits_into
    USERS ||--o{ CONVERSATIONS : starts
    WORKSPACES ||--o{ CONVERSATIONS : may_have
    CONVERSATIONS ||--o{ MESSAGES : contains

    USERS {
      int id PK
      string username
      string email
      string password_hash
      datetime created_at
      datetime updated_at
    }

    WORKSPACES {
      int id PK
      int user_id FK
      string name
      datetime created_at
      datetime updated_at
    }

    DOCUMENTS {
      int id PK
      int workspace_id FK
      string title
      text content
      datetime created_at
      datetime updated_at
    }

    CHUNKS {
      int id PK
      int document_id FK
      text content
      int position
      datetime created_at
    }

    CONVERSATIONS {
      int id PK
      int user_id FK
      int workspace_id FK
      datetime started_at
      datetime ended_at
    }

    MESSAGES {
      int id PK
      int conversation_id FK
      int user_id FK
      text content
      datetime sent_at
    }
```

### Vector Database (Qdrant) Collections

- **Document Embeddings**
  - `id` (UUID)
  - `embedding` (vector)
  - `metadata` (JSON)
- **Chunk Embeddings**
  - `id` (UUID)
  - `document_id` (UUID)
  - `embedding` (vector)
  - `metadata` (JSON)

---

## 🛠️ Implementation Guide

<details open>
<summary><strong>1) Setup</strong></summary>

### Prerequisites
- Node.js
- Python
- Docker

### Bootstrap
```bash
git clone https://github.com/emxelux/ChatPDF.git
cd ChatPDF
```

### Frontend Install
```bash
cd frontend
npm install
```

### Backend Install
```bash
cd backend
pip install -r requirements.txt
```
</details>

<details>
<summary><strong>2) Dependencies</strong></summary>

- **Frontend:** React, Axios
- **Backend:** Flask, SQLAlchemy
</details>

<details>
<summary><strong>3) Configuration</strong></summary>

### Frontend `.env`
```env
REACT_APP_API_URL=http://localhost:5000/api
```

### Backend Config
Set database connection values inside `backend/config.py`.
</details>

<details>
<summary><strong>4) Code Patterns</strong></summary>

### Frontend API Call
```javascript
import axios from 'axios';

const fetchData = async () => {
  const response = await axios.get(`${process.env.REACT_APP_API_URL}/data`);
  console.log(response.data);
};
```

### Backend Endpoint
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({'data': 'Hello World!'})
```

### Document Ingestion
```python
with open('document.txt') as f:
    content = f.read()
    # Process content here
```

### Indexing (Example)
```bash
curl -X POST "http://localhost:9200/my_index/_doc/1" -H 'Content-Type: application/json' -d '{ "content": "Document content here" }'
```

### Retrieval (Example)
```bash
curl -X GET "http://localhost:9200/my_index/_search?q=document"
```

### Agent Skeleton
```python
class DataAgent:
    def query_data(self):
        # Query logic here
```
</details>

---

## 🔐 Security

- **Authentication** before system access.
- **Encryption** for data in transit and at rest.
- **Input validation** to mitigate injection and malformed payload attacks.

---

## 📈 Scalability Notes

- Index foreign keys in PostgreSQL for better relational query performance.
- Use vector indexes in Qdrant for low-latency semantic retrieval.
- Decouple ingestion and retrieval services for horizontal scaling.
- Keep chat history and metadata structured for auditability and personalization.

---

## ✅ Why This Project Stands Out

- End-to-end RAG blueprint (UI → API → Retrieval → Generation).
- Production-minded data model spanning relational + vector storage.
- Clear onboarding path with setup and implementation snippets.
- Visual-first documentation with architecture, wireframes, and workflow diagrams.

---

<div align="center">

### Built for modern AI product engineering interviews and real-world delivery.

</div>
=======
---
title: Chatpdf
emoji: ⚡
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
short_description: This is my AI Rag system project.
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 8295253039b6cbfb540bd372bafb2427af962082
