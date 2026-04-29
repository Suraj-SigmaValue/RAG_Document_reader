import os
import tempfile
from typing import List, Optional, TypedDict

import tiktoken
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
POPPLER_PATH = os.getenv("POPPLER_PATH", "")
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

OCR_AVAILABLE = True
try:
    from pdf2image import convert_from_path
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except ImportError:
    OCR_AVAILABLE = False


class RagRuntime:
    def __init__(self) -> None:
        self.vectorstore: Optional[FAISS] = None
        self.retriever = None
        self.document_name: Optional[str] = None
        self.chunk_count = 0
        self.total_embedding_tokens = 0
        self.total_llm_input_tokens = 0
        self.total_llm_output_tokens = 0


runtime = RagRuntime()
app = FastAPI(title="RAG Document Reader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
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
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing. Add it to your .env file.",
        )


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def count_tokens_for_chunks(chunks: List[Document], model: str = "text-embedding-ada-002") -> int:
    return sum(count_tokens(chunk.page_content, model) for chunk in chunks)


def preprocess_for_bm25(text: str) -> List[str]:
    return text.lower().split()


def load_pdf_with_ocr(file_path: str) -> List[Document]:
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="OCR libraries are not installed. Install pdf2image and pytesseract.",
        )

    try:
        if POPPLER_PATH:
            images = convert_from_path(file_path, dpi=300, poppler_path=POPPLER_PATH)
        else:
            images = convert_from_path(file_path, dpi=300)

        documents = []
        for index, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": index + 1, "source": "ocr"},
                    )
                )
        return documents
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {exc}. Poppler and Tesseract must be installed.",
        ) from exc


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


def build_retriever(chunks: List[Document]):
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 40,
            "lambda_mult": 0.5,
        },
    )

    sparse_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=preprocess_for_bm25,
    )
    sparse_retriever.k = 10

    hybrid_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.6, 0.4],
    )

    return vectorstore, hybrid_retriever


def retrieve_node(state: GraphState) -> GraphState:
    if runtime.retriever is None:
        return {**state, "answer": "Please upload and process a document first."}

    question = state["question"]
    expanded_query = f"""
    {question}

    Search specifically for definitions, clauses, regulation numbers, named terms,
    calculation rules, tables, and exact phrases that may answer the question.
    """

    docs = runtime.retriever.invoke(expanded_query)
    context_blocks = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", runtime.document_name or "document")
        context_blocks.append({
            "source": str(source),
            "page": str(page),
            "content": doc.page_content
        })

    return {**state, "context": context_blocks}


def generate_node(state: GraphState) -> GraphState:
    if not state["context"]:
        return {**state, "answer": "No relevant content found in the document."}

    context_str = "\n\n---\n\n".join(
        [f"[Source: {c['source']}, Page: {c['page']}]\n{c['content']}" for c in state["context"]]
    )
    prompt = f"""You are an intelligent assistant that answers user questions primarily based on the provided document context.

Guidelines:
1. Use the provided context as the main source of truth.
2. If the answer is fully available in the context, respond confidently and clearly.
3. If the context provides partial information, answer what is supported and say what is missing.
4. Do not introduce unsupported facts.
5. If the answer is not found, say: "I don't have enough information to answer that."
6. Make sure Answer is concise, complete, and directly addresses the question."

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
        "name": "RAG Document Reader API",
        "status": "running",
        "frontend": "http://localhost:3000",
        "docs": "http://localhost:8080/docs",
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

        embedding_tokens = count_tokens_for_chunks(chunks)
        vectorstore, retriever = build_retriever(chunks)

        runtime.vectorstore = vectorstore
        runtime.retriever = retriever
        runtime.document_name = file.filename
        runtime.chunk_count = len(chunks)
        runtime.total_embedding_tokens += embedding_tokens

        return {
            "document_name": runtime.document_name,
            "pages_or_sections": len(documents),
            "chunk_count": runtime.chunk_count,
            "message": "Document indexed with hybrid retrieval.",
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

    if runtime.retriever is None:
        raise HTTPException(status_code=400, detail="Upload and process a document first.")

    final_state = rag_graph.invoke({"question": question, "context": [], "answer": ""})
    return AskResponse(
        answer=final_state["answer"],
        chunks=final_state["context"],
        token_usage=token_usage(),
    )
