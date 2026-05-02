import os
import tempfile
from typing import List, Optional, TypedDict
import re
from langchain_core.documents import Document
import fitz  # PyMuPDF
import base64

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
        self.page_images = {}
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

def extract_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    image_docs = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")

            # Convert to base64 (for UI display)
            encoded = base64.b64encode(image_bytes).decode("utf-8")

            image_docs.append(
                Document(
                    page_content="[IMAGE]",
                    metadata={
                        "page": page_index + 1,
                        "image_base64": encoded,
                        "image_mime": f"image/{image_ext}",
                        "image_index": img_index,
                        "type": "image"
                    }
                )
            )

    return image_docs


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
            for doc in docs:
                metadata = doc.metadata or {}   # ✅ MUST come first
                page = doc.metadata.get("page")
                if isinstance(page, int):
                    doc.metadata["page"] = page + 1
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

def is_table_like(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if len(lines) < 3:
        return False

    table_keywords = [
        "table no", "sr. no", "occupancy"
    ]

    keyword_hit = any(k in text.lower() for k in table_keywords)

    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    numeric_ratio = numeric_lines / len(lines)

    column_like_lines = sum(
        1 for line in lines
        if len(re.split(r"\s{2,}|\t", line)) >= 2
    )

    return keyword_hit or numeric_ratio > 0.35 or column_like_lines >= 3

def is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    toc_line_count = 0
    for line in lines:
        has_section = re.match(r"^\d+(?:\.\d+)*\s+", line)
        ends_with_page = re.search(r"(?:\.{2,}\s*|\s+)\d{1,4}$", line)
        if has_section and ends_with_page:
            toc_line_count += 1

    return (toc_line_count / len(lines)) >= 0.5


def section_roots_from_toc(docs: List[Document]) -> List[str]:
    section_ids = []

    for doc in docs:
        if not is_toc_like(doc.page_content):
            continue

        section_ids.extend(re.findall(r"\b(\d+(?:\.\d+){1,4})\b", doc.page_content))

    roots = []
    for section_id in section_ids:
        parts = section_id.split(".")
        root = ".".join(parts[:2]) if len(parts) > 2 else section_id
        if root not in roots:
            roots.append(root)

    return roots


def expand_section_docs_from_toc(docs: List[Document], max_extra: int = 18) -> List[Document]:
    roots = section_roots_from_toc(docs)
    if not roots:
        return []

    expanded_docs = []
    seen = set()

    for chunk in runtime.chunks:
        section = str(chunk.metadata.get("section") or "")
        if not section:
            continue

        matches_root = any(section == root or section.startswith(f"{root}.") for root in roots)
        if not matches_root or is_toc_like(chunk.page_content):
            continue

        key = (chunk.metadata.get("page"), section, chunk.page_content[:120])
        if key in seen:
            continue

        expanded_docs.append(chunk)
        seen.add(key)

        if len(expanded_docs) >= max_extra:
            break

    return expanded_docs


def structure_based_split(documents):
    structured_chunks = []

    # section_pattern = r"\n(?=\d+(?:\.\d+)*)"
    section_pattern = r"\n(?=\d+\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{4,80})"

    for doc in documents:
        if doc.metadata.get("type") == "image":
            structured_chunks.append(doc)
            continue

        text = doc.page_content
        page = doc.metadata.get("page", None)

        # Split by sections like 1., 1.1, 4.3 etc.
        sections = re.split(section_pattern, text)

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            # Extract section number/title
            match = re.match(r"(\d+(\.\d+)*)\s*(.*)", sec)

            section_id = match.group(1) if match else None
            title = match.group(3)[:100] if match else None

            structured_chunks.append(
                Document(
                    page_content=sec,
                    metadata={
                        "page": page,
                        "source": doc.metadata.get("source", runtime.document_name or "document"),
                        "section": section_id,
                        "title": title,
                        "chunk_type": "section",
                        "is_table": is_table_like(sec) 
                    }
                )
            )

    return structured_chunks

def hybrid_chunking(documents):
    structured_docs = structure_based_split(documents)
    final_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", "; ", " ", ""],
    )

    for doc in structured_docs:
        if doc.metadata.get("type") == "image":
            final_chunks.append(doc)
            continue

        if len(doc.page_content) > 2000:
            sub_chunks = text_splitter.split_documents([doc])
            for i, sub in enumerate(sub_chunks):
                sub.metadata.update(doc.metadata)
                sub.metadata["chunk_type"] = "sub_chunk"
                sub.metadata["sub_chunk_index"] = i
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)

    return final_chunks

def retrieve_node(state: GraphState) -> GraphState:
    if runtime.faiss_index is None:
        return {**state, "answer": "Please upload and process a document first."}

    question = state["question"]

    expanded_query = f"""
    {question}

    Search specifically for:
    - definitions
    - clauses and regulations
    - tables and structured data
    - figures, diagrams, and images
    - exact phrases and references
    """

    docs = runtime.ensemble_retriever(expanded_query, k=10)
    expanded_section_docs = expand_section_docs_from_toc(docs)
    if expanded_section_docs:
        docs = [doc for doc in docs if not is_toc_like(doc.page_content)] + expanded_section_docs

    context_blocks = []
    retrieved_pages = []

    for doc in docs:
        metadata = doc.metadata or {}   

        page = metadata.get("page", "unknown")
        source = metadata.get("source", runtime.document_name or "document")
        score = metadata.get("hybrid_score", 0)
        if page != "unknown":
            retrieved_pages.append(page)

            # ✅ NOW safe to use metadata
        if metadata.get("is_table"):
            score += 0.2   # boost table chunks

        # 🔥 CASE 1: IMAGE HANDLING
        if metadata.get("type") == "image":
            context_blocks.append({
                "source": str(source),
                "page": str(page),
                "type": "image",
                "image_base64": metadata.get("image_base64"),
                "image_mime": metadata.get("image_mime", "image/png"),
                "relevance_score": score
            })

        # 🔥 CASE 2: TABLE HANDLING
        elif metadata.get("is_table"):
            context_blocks.append({
                "source": str(source),
                "page": str(page),
                "type": "table",
                "content": doc.page_content,
                "relevance_score": score
            })

        # 🔥 CASE 3: NORMAL TEXT
        else:
            context_blocks.append({
                "source": str(source),
                "page": str(page),
                "type": "text",
                "content": doc.page_content,
                "relevance_score": score
            })

    image_keys_added = set()

    top_pages = set()

    # only take top 3 most relevant chunks
    for doc in docs[:3]:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            top_pages.add(page)

    for page_key in top_pages:
        try:
            page_key = int(page)
        except (TypeError, ValueError):
            continue

        for image_doc in runtime.page_images.get(page_key, []):
            metadata = image_doc.metadata or {}
            image_key = (page_key, metadata.get("image_index", 0), metadata.get("image_base64", "")[:32])
            if image_key in image_keys_added:
                continue

            context_blocks.append({
                "source": str(metadata.get("source", runtime.document_name or "document")),
                "page": str(page_key),
                "type": "image",
                "image_base64": metadata.get("image_base64"),
                "image_mime": metadata.get("image_mime", "image/png"),
                "relevance_score": metadata.get("hybrid_score", 0),
            })
            image_keys_added.add(image_key)

            if len(image_keys_added) >= 6:
                break

        if len(image_keys_added) >= 6:
            break

    return {**state, "context": context_blocks}


def generate_node(state: GraphState) -> GraphState:
    if not state["context"]:
        return {**state, "answer": "No relevant content found in the document."}

    context_parts = []

    for c in state["context"]:
        if c.get("type") == "image":
            context_parts.append(
                f"[Source: {c['source']}, Page: {c['page']}]\n"
                "[IMAGE available for UI rendering]"
            )

        elif c.get("type") == "table":
            context_parts.append(
                f"[Source: {c['source']}, Page: {c['page']}]\n"
                f"[TABLE]\n{c['content']}"
            )

        else:
            context_parts.append(
                f"[Source: {c['source']}, Page: {c['page']}]\n"
                f"{c['content']}"
            )

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an intelligent assistant that answers user queries strictly using the provided document context as the primary source of truth.

========================
CORE PRINCIPLES
========================

1. USE ONLY CONTEXT  
- Do not assume or infer missing information  
- If the answer is not present, respond with:  
  "I don't have enough information in the document to answer that."

2. FAITHFULNESS  
- Do NOT modify, reinterpret, or summarize critical content  
- Preserve original meaning, values, structure, and wording  
- Do NOT hallucinate or fill gaps  

3. COMPLETENESS  
- If the relevant content is long, return it fully  
- Do NOT shorten or compress important sections  
- Include all necessary parts for clarity  

========================
STRUCTURE HANDLING (VERY IMPORTANT)
========================

You must preserve and reconstruct the structure of the document wherever applicable:

1. TABLES  
- If any part of the context contains tabular or semi-tabular data:
  → Reconstruct it into a clean table format  
  → Preserve all rows and columns  
  → Do NOT summarize or convert into paragraphs  
  → If multiple tables exist, present each separately  
  → Include table title (if available)  

2. SECTIONS & CLAUSES  
- Maintain hierarchy (section, subsection, clause)  
- Clearly label headings if present  

3. LISTS  
- Preserve bullet points or numbered lists exactly  

========================
VISUAL CONTENT HANDLING (CRITICAL)
========================

If the answer to the user’s query includes or is supported by:

- diagrams  
- figures  
- images  
- charts  
- graphical representations  

THEN:

- The UI renders retrieved image data directly from the response payload.
- Do not invent Markdown image URLs, broken image placeholders, or external image references.
- ALWAYS include the visual content in the output if it exists in the context  
- NEVER skip or ignore visual elements  
- If image data (e.g., base64 or reference) is present → include it explicitly  
- If the image cannot be rendered → provide a reference or placeholder indicating its presence  
- Ensure visual content is presented alongside the relevant textual explanation  

IMPORTANT RULE:  
If a diagram, table, or visual element is part of the answer, it is MANDATORY to include it.  
Do NOT provide text-only answers when visual support exists.
add page numbers, source references, and clearly associate them with the content they support.

========================
PAGE & SOURCE INFORMATION
========================

- Always include page numbers if available  
- Mention source identifiers if present  
- Keep references clearly associated with content  

========================
OUTPUT RULES
========================

- Prefer structured output over narrative text  
- Use headings, tables, and lists where applicable  
- Do NOT mix unrelated sections  
- Do NOT generate generic summaries when structured or visual data exists  
- If multiple relevant sections exist, present all clearly  

========================
DECISION LOGIC
========================

Follow this order:

1. Find exact match in context  
2. If structured data exists → return structured (table/list/section)  
3. If visual content exists → include it  
4. If unstructured text → return complete relevant portion  
5. If partial info → return available + clearly mention missing  
6. If no info → say "I don't have enough information in the document"

========================
STYLE
========================

- Clear, formal, and readable  
- Structured > descriptive  
- Complete > summarized  
- Visual + text combination preferred when available  
 

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
        for document in documents:
            document.metadata["source"] = file.filename
        image_docs = extract_images_from_pdf(tmp_path) if suffix.lower() == ".pdf" else []
        runtime.page_images = {}
        for image_doc in image_docs:
            image_doc.metadata["source"] = file.filename
            page = image_doc.metadata.get("page")
            if isinstance(page, int):
                runtime.page_images.setdefault(page, []).append(image_doc)
        if not documents:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1800,
        #     chunk_overlap=350,
        #     separators=["\n\n", "\n", ". ", "; ", " ", ""],
        # )
        # chunks = text_splitter.split_documents(documents)
        chunks = hybrid_chunking(documents)

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
