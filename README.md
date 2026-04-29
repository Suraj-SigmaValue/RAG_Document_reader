
# RAG Document Reader

## What Is This System?
This is a lightweight Retrieval-Augmented Generation (RAG) app that allows users to upload and query documents (PDF, DOCX) using natural language. It combines a FastAPI backend for document processing and a Next.js 14 frontend for a modern user interface.

## Why This System Exists
To make it easy for users to extract, search, and understand information from complex documents using state-of-the-art retrieval and LLM techniques, with a focus on transparency and accuracy.

## Who Is This Most Useful For?
- Legal professionals
- Compliance officers
- Researchers
- Anyone needing to search and interpret large or complex documents

## Current Progress
- Document ingestion and OCR for PDF/DOCX
- Hybrid retrieval (semantic + keyword)
- Knowledge graph-based query flow
- Modern frontend with source highlighting

## Tech Stack
- **Backend:** FastAPI, LangChain, FAISS, OpenAI Embeddings, BM25, PyPDFLoader, pdf2image, pytesseract
- **Frontend:** Next.js 14, TypeScript, Tailwind CSS
- **Other:** StateGraph (LangGraph), glassmorphism UI

## 5. Workflow and Execution
1. User uploads a document (PDF/DOCX)
2. Backend ingests and chunks the document
3. Hybrid retrieval (semantic + keyword) finds relevant context
4. LLM generates an answer with source references
5. Frontend displays answer and source cards

## Stage 2 - Planning
- Define user stories and document types
- Design chunking and retrieval strategies
- Plan UI/UX for document interaction

## Stage 3 - Tool Selection
- Select OCR, embedding, and retrieval libraries
- Choose LLM provider and frontend framework

## In document
All references to 'Document in' have been corrected to 'document'.

## Backend

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8080
```

The API runs at `http://localhost:8080`.

## Frontend

```powershell
cd frontend
npm install
npm run dev
```

The app runs at `http://localhost:3000`.

Set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local` if your backend uses a different URL.
