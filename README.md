# RAG Document Reader

A local full-stack document question-answering application for PDF and DOCX files. The system combines document ingestion, OCR fallback, embedded PDF image extraction, hybrid FAISS/BM25 retrieval, and OpenAI-powered answer generation behind a FastAPI backend and a Next.js frontend.

## Overview

RAG Document Reader helps users upload complex documents and ask natural-language questions about their contents. It is designed for regulatory, legal, compliance, research, and technical documents where answers must remain grounded in the uploaded source material.

The application retrieves the most relevant document chunks using both semantic search and keyword search, sends the selected context to an LLM, and displays the final answer with retrieved source chunks, supporting images, and cumulative token usage for review.

## Key Capabilities

- Upload and process PDF and DOCX documents.
- Extract text from digital PDFs using `PyPDFLoader`.
- Fall back to OCR for scanned PDFs using `pdf2image` and `pytesseract`.
- Extract embedded PDF images using PyMuPDF and return them to the UI as base64 image chunks.
- Load DOCX content using LangChain's DOCX loader.
- Split documents using section-aware chunking for regulation-style documents.
- Detect table-like text and table-of-contents-like sections to improve retrieval context.
- Build a FAISS vector index using OpenAI embeddings.
- Combine FAISS semantic retrieval with BM25 keyword retrieval.
- Use reciprocal rank fusion to merge dense and sparse results.
- Generate document-grounded answers through a LangGraph workflow.
- Display answers in a clean report-style UI.
- Render Markdown answers with `react-markdown` and GitHub Flavored Markdown support.
- Render valid Markdown tables as structured HTML tables.
- Render retrieved PDF images as supporting image cards.
- Show retrieved source chunks for transparency.
- Track cumulative LLM token usage.

## Architecture

The project is split into two main applications:

- `backend/`: FastAPI service for ingestion, retrieval, and answer generation.
- `frontend/`: Next.js 14 application for document upload, questions, answers, and source review.

High-level workflow:

1. The user uploads a PDF or DOCX file from the frontend.
2. The backend saves the file temporarily and extracts text.
3. Extracted text is split into structured chunks.
4. OpenAI embeddings are generated for each chunk.
5. A FAISS index is created for semantic retrieval.
6. A BM25 retriever is created for keyword retrieval.
7. The user submits a question.
8. The backend expands the query and retrieves relevant chunks using hybrid retrieval.
9. Table-like chunks and related section chunks are emphasized where available.
10. LangGraph passes the retrieved context to the answer generation node.
11. The LLM returns a grounded, formatted answer.
12. The frontend renders the answer, tables, supporting images, token usage, and source chunks.

## Technology Stack

### Backend

- Python
- FastAPI
- LangChain
- LangGraph
- OpenAI chat models and embeddings
- FAISS
- BM25 / `rank-bm25`
- PyPDF / LangChain PDF loader
- PyMuPDF
- `pdf2image`
- `pytesseract`
- `python-dotenv`
- `tiktoken`

### Frontend

- Next.js 14
- React 18
- TypeScript
- `react-markdown`
- `remark-gfm`
- Plain CSS

## Project Structure

```text
.
|-- backend/
|   |-- __init__.py
|   |-- main.py
|   `-- Backup_with_Vector.py
|-- frontend/
|   |-- app/
|   |   |-- globals.css
|   |   |-- layout.tsx
|   |   `-- page.tsx
|   |-- next.config.mjs
|   |-- package.json
|   `-- tsconfig.json
|-- docx_render/
|-- build_documentation.py
|-- Documentation.docx
|-- Documentation.pdf
|-- requirements.txt
`-- README.md
```

## Prerequisites

Install the following before running the application:

- Python 3.10 or newer
- Node.js 18 or newer
- npm
- OpenAI API key
- Tesseract OCR, required only for scanned PDF OCR
- Poppler, required by `pdf2image` for OCR-based PDF conversion

## Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
POPPLER_PATH=C:\path\to\poppler\Library\bin
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

Notes:

- `OPENAI_API_KEY` is required.
- `POPPLER_PATH` can be left empty if Poppler is already available on the system path.
- `TESSERACT_CMD` should point to the local Tesseract executable on Windows.

If the backend runs on a URL other than `http://localhost:8080`, create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8080
```

## Backend Setup

From the project root:

```powershell
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8080
```

The backend will be available at:

```text
http://localhost:8080
```

## Frontend Setup

From the project root:

```powershell
cd frontend
npm install
npm run dev
```

The frontend will be available at:

```text
http://localhost:3000
```

## API Reference

### `GET /`

Returns API metadata and the available endpoint list.

Example response:

```json
{
  "name": "RAG Document Reader API with FAISS",
  "status": "running",
  "endpoints": {
    "health": "/health",
    "upload_document": "POST /documents",
    "ask_question": "POST /ask",
    "status": "/status"
  }
}
```

### `GET /health`

Returns a basic health check.

Example response:

```json
{
  "ok": true
}
```

### `GET /status`

Returns the currently indexed document state and token usage.

Example response:

```json
{
  "document_name": "example.pdf",
  "chunk_count": 42,
  "token_usage": {
    "input": 1200,
    "output": 350
  },
  "retrieval_method": "hybrid (FAISS + BM25)"
}
```

### `POST /documents`

Uploads and indexes a document.

Supported file types:

- `.pdf`
- `.docx`

Request type:

```text
multipart/form-data
```

Form field:

```text
file
```

Example response:

```json
{
  "document_name": "example.pdf",
  "pages_or_sections": 12,
  "chunk_count": 38,
  "message": "Document indexed with FAISS + BM25 hybrid retrieval",
  "token_usage": {
    "input": 0,
    "output": 0
  }
}
```

### `POST /ask`

Asks a question against the indexed document.

Request body:

```json
{
  "question": "What are the road width requirements?"
}
```

Example response:

```json
{
  "answer": "The answer generated from the retrieved document context.",
  "chunks": [
    {
      "source": "example.pdf",
      "page": "3",
      "type": "text",
      "content": "Relevant retrieved text...",
      "relevance_score": 0.0123
    }
  ],
  "token_usage": {
    "input": 2400,
    "output": 520
  }
}
```

## Retrieval and Answering Logic

The backend uses a hybrid retrieval pipeline:

- FAISS retrieves semantically similar chunks using OpenAI embeddings.
- BM25 retrieves chunks with strong keyword overlap.
- Reciprocal Rank Fusion combines the two result sets.
- The top combined chunks are passed to the generation step.
- Table-like chunks receive additional retrieval emphasis.
- If retrieved context appears to be a table of contents, related section chunks can be expanded into the answer context.

The answer generation prompt instructs the model to:

- Use only the retrieved document context.
- Avoid unsupported assumptions.
- Preserve exact values, dates, regulations, definitions, and tables.
- Render tables as valid Markdown tables.
- Include source and page references when available.
- Preserve and describe available visual content when image chunks are present.
- Clearly state when the context does not contain enough information.

## Frontend Rendering

The frontend uses `react-markdown` with `remark-gfm` for document answers:

- Headings, paragraphs, lists, and links are rendered as native HTML.
- Markdown tables are rendered through GitHub Flavored Markdown support.
- Image chunks returned by the backend are rendered as data URLs in a Supporting Images section.
- Retrieved source chunks are shown in an expandable details section.

For best table output, the backend prompt asks the model to produce tables using this format:

```markdown
| Column A | Column B |
| --- | --- |
| Value 1 | Value 2 |
```

## Common Usage Flow

1. Start the backend on port `8080`.
2. Start the frontend on port `3000`.
3. Open `http://localhost:3000`.
4. Upload a PDF or DOCX file.
5. Wait for the document to be processed.
6. Ask a specific question.
7. Review the answer and source chunks.

## Documentation Files

The repository includes generated project documentation:

- `Documentation.docx`
- `Documentation.pdf`

Use the documentation builder when the implementation changes:

```powershell
.\venv\Scripts\python.exe build_documentation.py
```

If either generated documentation file is open in Word, a browser PDF viewer, or an IDE preview, Windows may lock the file and prevent replacement. Close the open preview and rerun the command.

## Troubleshooting

### `OPENAI_API_KEY is missing`

Create or update the root `.env` file and ensure it contains a valid `OPENAI_API_KEY`.

### Backend cannot process scanned PDFs

Verify that Tesseract and Poppler are installed. Update `TESSERACT_CMD` and `POPPLER_PATH` in `.env` if needed.

### Frontend cannot connect to the API

Confirm the backend is running at `http://localhost:8080`. If using a different URL, set `NEXT_PUBLIC_API_BASE_URL` in `frontend/.env.local`.

### Tables appear as plain text

The UI renders valid Markdown tables. Ensure the generated answer includes a header row, separator row, and rows with matching pipe-delimited columns.

### Supporting images do not appear

Confirm the uploaded source is a PDF containing embedded images and that the relevant page or image chunk is retrieved for the question. DOCX image extraction is not implemented in the current backend.

### Answers are incomplete

Ask a more specific question or upload a document with clearer extractable text. For scanned PDFs, check whether OCR is installed and producing usable text.

## Development Notes

- The current runtime stores one indexed document in memory.
- Uploading a new document replaces the active in-memory index.
- Token usage is cumulative for the current backend process.
- FAISS and BM25 indexes are not persisted to disk.
- This project is intended as a local demo or prototype; production deployment should add authentication, file validation, persistent storage, logging, and stronger operational controls.

## Verification Commands

Run the frontend build:

```powershell
cd frontend
npm run build
```

Run the backend locally:

```powershell
uvicorn backend.main:app --reload --port 8080
```

Check API health:

```powershell
Invoke-RestMethod http://localhost:8080/health
```
