import os
import tempfile
from typing import List, Optional, TypedDict

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import faiss
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POPPLER_PATH = os.getenv("POPPLER_PATH", "")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_path
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except ImportError:
    OCR_AVAILABLE = False


class RagRuntime:
    def __init__(self) -> None:
        self.faiss_index = None  # FAISS index for similarity search
        self.chunks = []  # Store chunks for retrieval
        self.bm25_retriever = None
        self.document_name: Optional[str] = None
        self.chunk_count = 0
        self.total_llm_input_tokens = 0
        self.total_llm_output_tokens = 0
        self.embeddings = None


runtime = RagRuntime()
app = FastAPI(title="RAG Document Reader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    chunks: List[dict]
    token_usage: dict


class GraphState(TypedDict):
    question: str
    context: List[dict]
    answer: str


def require_openai_key() -> None:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def preprocess_for_bm25(text: str) -> List[str]:
    return text.lower().split()


def load_pdf_with_ocr(file_path: str) -> List[Document]:
    if not OCR_AVAILABLE:
        raise HTTPException(status_code=500, detail="OCR libraries not installed")

    try:
        images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH if POPPLER_PATH else None)
        documents = []
        for index, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                documents.append(Document(page_content=text, metadata={"page": index + 1, "source": "ocr"}))
        return documents
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")


def load_documents(file_path: str, filename: str) -> List[Document]:
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            total_text = " ".join(doc.page_content for doc in docs)
            if len(total_text.strip()) > 500:
                return docs
        except Exception:
            pass
        return load_pdf_with_ocr(file_path)

    if extension == ".docx":
        return Docx2txtLoader(file_path).load()

    raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")


def create_faiss_retriever(chunks: List[Document]):
    """Create FAISS index for fast similarity search"""
    
    # Initialize OpenAI embeddings
    runtime.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
    
    # Generate embeddings for all chunks
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = runtime.embeddings.embed_documents(chunk_texts)
    
    # Store chunks for retrieval
    runtime.chunks = chunks
    
    # Convert embeddings to numpy array
    embeddings_array = np.array(chunk_embeddings).astype('float32')
    
    # Create FAISS index (using inner product for cosine similarity)
    dimension = len(embeddings_array[0])
    index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity after normalization)
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)
    
    runtime.faiss_index = index
    return index


def create_hybrid_retriever(chunks: List[Document], vector_weight: float = 0.6, bm25_weight: float = 0.4):
    """Create hybrid retriever combining FAISS (dense) and BM25 (sparse) search"""
    
    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=preprocess_for_bm25)
    bm25_retriever.k = 10
    runtime.bm25_retriever = bm25_retriever
    
    def hybrid_search(query: str, k: int = 10) -> List[Document]:
        # Get BM25 results
        bm25_docs = bm25_retriever.invoke(query)
        
        # Generate query embedding and search with FAISS
        query_embedding = runtime.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        # Search FAISS index
        distances, indices = runtime.faiss_index.search(query_array, k * 2)
        
        # Get FAISS results
        faiss_docs = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(runtime.chunks):
                doc = runtime.chunks[idx]
                doc.metadata["faiss_score"] = float(distance)
                doc.metadata["faiss_rank"] = len(faiss_docs) + 1
                faiss_docs.append(doc)
        
        # Reciprocal Rank Fusion
        scores = {}
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content[:100]
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rank + 60)
            doc.metadata["bm25_rank"] = rank + 1
        
        # Add FAISS scores
        for rank, doc in enumerate(faiss_docs):
            doc_id = doc.page_content[:100]
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (rank + 60)
        
        # Combine and sort
        all_docs = bm25_docs + faiss_docs
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
                unique_docs[doc_id].metadata["hybrid_score"] = scores[doc_id]
        
        sorted_docs = sorted(unique_docs.values(), key=lambda x: x.metadata.get("hybrid_score", 0), reverse=True)
        return sorted_docs[:k]
    
    return hybrid_search


def retrieve_node(state: GraphState) -> GraphState:
    if runtime.faiss_index is None:
        return {**state, "answer": "Please upload and process a document first."}

    question = state["question"]
    expanded_query = f"""
    {question}

    Search specifically for definitions, clauses, regulation numbers, named terms,
    calculation rules, tables, and exact phrases that may answer the question.
    """
    
    docs = runtime.ensemble_retriever(expanded_query, k=10)
    
    context_blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", runtime.document_name or "document")
        context_blocks.append({
            "source": str(source),
            "page": str(page),
            "content": doc.page_content,
            "relevance_score": doc.metadata.get("hybrid_score", 0)
        })
    
    return {**state, "context": context_blocks}


def generate_node(state: GraphState) -> GraphState:
    if not state["context"]:
        return {**state, "answer": "No relevant content found in the document."}

    context_str = "\n\n---\n\n".join(
        [f"[Source: {c['source']}, Page: {c['page']}]\n{c['content']}" for c in state["context"]]
    )
    
    prompt = f"""You are an intelligent assistant that answers user queries using the provided document context as the primary source of truth.

### Core Behavior
- Base your answer strictly on the provided context.
- Do not use external knowledge or assumptions.
- Ensure the response is accurate, clear, and professionally written.

### Answer Quality
- Present the answer in a well-structured format (use headings, bullet points, or sections where appropriate).
- Use clear and formal language, especially for regulatory, legal, or technical topics.
- Avoid vague or conversational responses.

### Completeness Handling
- If the answer is fully available in the context:
  → Provide a complete and confident response.

- If the answer is partially available:
  → Provide all available details.
  → Clearly state what information is missing from the context.

- If the answer is not available:
  → Respond with: "I don't have enough information to answer that."

### Faithfulness Rules
- Do not hallucinate or infer missing details.
- Do not expand beyond what is explicitly supported.
- Do not modify numerical values, percentages, dates, or definitions.

### Clarity Enhancements
- When listing schemes, regulations, or components:
  → Break them into numbered or titled sections.
- When definitions are present:
  → Present them clearly and distinctly.
- When multiple components exist:
  → Explain each component separately.

### Output Style
- Keep the response concise but complete.
- Prioritize readability and logical flow."

Context:
{context_str}

Question: {state["question"]}

Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)

    usage = response.response_metadata.get("token_usage", {})
    if usage:
        runtime.total_llm_input_tokens += usage.get("prompt_tokens", 0)
        runtime.total_llm_output_tokens += usage.get("completion_tokens", 0)
    else:
        runtime.total_llm_input_tokens += count_tokens(prompt, "gpt-4o-mini")
        runtime.total_llm_output_tokens += count_tokens(response.content, "gpt-4o-mini")

    return {**state, "answer": response.content}


builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
rag_graph = builder.compile()


def token_usage() -> dict:
    return {
        "input": runtime.total_llm_input_tokens,
        "output": runtime.total_llm_output_tokens,
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/")
def root() -> dict:
    return {
        "name": "RAG Document Reader API with FAISS",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload_document": "POST /documents",
            "ask_question": "POST /ask",
            "status": "/status",
        },
    }


@app.get("/status")
def status() -> dict:
    return {
        "document_name": runtime.document_name,
        "chunk_count": runtime.chunk_count,
        "token_usage": token_usage(),
        "retrieval_method": "hybrid (FAISS + BM25)"
    }


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    require_openai_key()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name.")

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        documents = load_documents(tmp_path, file.filename)
        if not documents:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=350,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document did not produce any text chunks.")

        # Create FAISS index
        create_faiss_retriever(chunks)
        
        # Create hybrid retriever
        runtime.ensemble_retriever = create_hybrid_retriever(chunks)
        
        runtime.document_name = file.filename
        runtime.chunk_count = len(chunks)

        return {
            "document_name": runtime.document_name,
            "pages_or_sections": len(documents),
            "chunk_count": runtime.chunk_count,
            "message": "Document indexed with FAISS + BM25 hybrid retrieval",
            "token_usage": token_usage(),
        }
    finally:
        os.unlink(tmp_path)


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    require_openai_key()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if runtime.faiss_index is None:
        raise HTTPException(status_code=400, detail="Upload and process a document first.")

    final_state = rag_graph.invoke({"question": question, "context": [], "answer": ""})
    return AskResponse(
        answer=final_state["answer"],
        chunks=final_state["context"],
        token_usage=token_usage(),
    )