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
from fastapi.responses import StreamingResponse
import json
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from pydantic import BaseModel
import faiss
import numpy as np
import time

from backend.prompt import RAG_PROMPT_TEMPLATE

load_dotenv()

RETRIEVAL_FAISS_K = 20
RETRIEVAL_BM25_K = 20
HYBRID_CANDIDATE_K = 40
RERANK_TOP_K = 15
PARENT_EXPAND_TOP_K = 6
PARENT_EXPAND_MAX_EXTRA = 15
MAX_CONTEXT_CHARS = 25000
MAX_IMAGES = 4
IMAGE_TOP_PAGES = 2

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
        self.loader_type: Optional[str] = None


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
    retrieval_timing: Optional[dict] = None


class GraphState(TypedDict):
    question: str
    context: List[dict]
    answer: str
    retrieval_timing: Optional[dict]
    token_usage: Optional[dict]

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


def load_pdf_with_opendataloader(file_path: str) -> List[Document]:
    import opendataloader_pdf
    import json
    
    result = opendataloader_pdf.convert(file_path, format="markdown,json")
    
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except Exception:
            data = []
    else:
        data = result
        
    docs = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "elements" in data:
        items = data["elements"]
    elif isinstance(data, dict) and "pages" in data:
        items = data["pages"]
    elif isinstance(data, dict):
        items = [data]
    else:
        items = []
        
    for item in items:
        md_content = item.get("markdown") or item.get("text") or item.get("content") or ""
        if not md_content:
            continue
            
        page_num = item.get("page", item.get("page_number", 1))
        
        docs.append(Document(
            page_content=md_content,
            metadata={
                "page": page_num,
                "source": "opendataloader",
                "type": "text",
                "section": item.get("section_id") or item.get("section"),
                "title": item.get("section_title") or item.get("title"),
                "bbox": item.get("bbox") or item.get("bounding_box"),
                "is_table": is_table_like(md_content)
            }
        ))
        
    return docs


def load_documents(file_path: str, filename: str) -> List[Document]:
    extension = os.path.splitext(filename)[1].lower()

    if extension == ".pdf":
        try:
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                page = doc.metadata.get("page")
                if isinstance(page, int):
                    doc.metadata["page"] = page + 1

            total_text = " ".join(doc.page_content for doc in docs)
            if len(total_text.strip()) > 500:
                runtime.loader_type = "pypdf"
                return docs

        except Exception as e:
            print(f"PyPDFLoader failed: {e}")

        try:
            docs = load_pdf_with_opendataloader(file_path)
            total_text = " ".join(doc.page_content for doc in docs)
            if len(total_text.strip()) > 500:
                runtime.loader_type = "opendataloader"
                return docs

        except Exception as e:
            print(f"OpenDataLoader failed, skipped safely: {e}")

        runtime.loader_type = "ocr"
        return load_pdf_with_ocr(file_path)

    if extension == ".docx":
        runtime.loader_type = "docx"
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
    bm25_retriever.k = RETRIEVAL_BM25_K
    runtime.bm25_retriever = bm25_retriever
    
    def hybrid_search(query: str, k: int = HYBRID_CANDIDATE_K) -> List[Document]:
        # Get BM25 results
        bm25_docs = bm25_retriever.invoke(query)
        
        # Generate query embedding and search with FAISS
        query_embedding = runtime.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_array)
        
        # Search FAISS index
        distances, indices = runtime.faiss_index.search(query_array, RETRIEVAL_FAISS_K)
        
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

    markdown_table = False
    for i, line in enumerate(lines):
        if "|" in line and i + 1 < len(lines):
            next_line = lines[i+1]
            if "|" in next_line and "---" in next_line:
                markdown_table = True
                break

    table_keywords = [
        "table", "schedule", "statement", "area", "rate", "cost", "amount", 
        "fsi", "carpet", "built-up", "premium", "charges", "sr no", 
        "description", "occupancy", "regulation"
    ]

    keyword_hit = any(k in text.lower() for k in table_keywords)

    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    numeric_ratio = numeric_lines / len(lines)

    column_like_lines = sum(
        1 for line in lines
        if len(re.split(r"\s{2,}|\t", line)) >= 2
    )

    return markdown_table or keyword_hit or numeric_ratio > 0.35 or column_like_lines >= 3

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


def rerank_documents(question: str, docs: List[Document], max_docs: int = RERANK_TOP_K) -> List[Document]:
    q_lower = question.lower()
    q_words = set(re.findall(r'\w+', q_lower))
    
    table_keywords = {"table", "rate", "area", "cost", "fsi", "calculation", "statement", "schedule"}
    has_table_intent = any(kw in q_words for kw in table_keywords)

    for doc in docs:
        score = doc.metadata.get("hybrid_score", 0)
        content_lower = doc.page_content.lower()
        title_lower = str(doc.metadata.get("title", "")).lower()
        section_lower = str(doc.metadata.get("section", "")).lower()
        
        content_words = set(re.findall(r'\w+', content_lower))
        if content_words:
            overlap = len(q_words & content_words)
            score += overlap * 0.05
        
        if any(qw in title_lower or qw in section_lower for qw in q_words if len(qw) > 3):
            score += 0.3
            
        if has_table_intent and doc.metadata.get("is_table"):
            score += 0.5
            
        if doc.metadata.get("page") is not None:
            score += 0.1
        if doc.metadata.get("section") is not None:
            score += 0.1
            
        if is_toc_like(doc.page_content):
            score -= 0.5
            
        doc.metadata["rerank_score"] = score
        
    reranked = sorted(docs, key=lambda x: x.metadata.get("rerank_score", 0), reverse=True)
    return reranked[:max_docs]


def expand_parent_sections(reranked_docs: List[Document]) -> List[Document]:
    expanded = list(reranked_docs)
    seen_ids = {doc.page_content[:100] for doc in expanded}
    
    extra_added = 0
    
    for doc in reranked_docs[:PARENT_EXPAND_TOP_K]:
        if extra_added >= PARENT_EXPAND_MAX_EXTRA:
            break
            
        if len(doc.page_content) > 1500:
            continue
            
        section = doc.metadata.get("section")
        parent_section = doc.metadata.get("parent_section")
        
        target_sections = set()
        if section: target_sections.add(section)
        if parent_section: target_sections.add(parent_section)
        
        if not target_sections:
            continue
            
        for chunk in runtime.chunks:
            if extra_added >= PARENT_EXPAND_MAX_EXTRA:
                break
                
            chunk_sec = chunk.metadata.get("section")
            chunk_parent = chunk.metadata.get("parent_section")
            
            if chunk_sec in target_sections or chunk_parent in target_sections:
                chunk_id = chunk.page_content[:100]
                if chunk_id not in seen_ids:
                    expanded.append(chunk)
                    seen_ids.add(chunk_id)
                    extra_added += 1
                    
    return expanded


def compress_context(question: str, docs: List[Document]) -> List[Document]:
    compressed_docs = []
    q_words = set(re.findall(r'\w+', question.lower()))
    
    total_chars = 0
    
    for doc in docs:
        if total_chars >= MAX_CONTEXT_CHARS:
            break
            
        if doc.metadata.get("is_table") or doc.metadata.get("type") == "image":
            compressed_content = doc.page_content
        else:
            paragraphs = doc.page_content.split("\n\n")
            kept_paragraphs = []
            for p in paragraphs:
                p_words = set(re.findall(r'\w+', p.lower()))
                if len(q_words & p_words) > 0 or len(p) < 50:
                    kept_paragraphs.append(p)
                    
            compressed_content = "\n\n".join(kept_paragraphs)
            if len(compressed_content) < len(doc.page_content) * 0.2:
                compressed_content = doc.page_content
                
        new_doc = Document(page_content=compressed_content, metadata=doc.metadata.copy())
        compressed_docs.append(new_doc)
        total_chars += len(compressed_content)
        
    return compressed_docs


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

            parent_section = None
            if section_id and "." in section_id:
                parts = section_id.split(".")
                parent_section = ".".join(parts[:-1])

            structured_chunks.append(
                Document(
                    page_content=sec,
                    metadata={
                        "page": page,
                        "source": doc.metadata.get("source", runtime.document_name or "document"),
                        "section": section_id,
                        "parent_section": parent_section,
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
                sub.metadata["parent_section"] = doc.metadata.get("parent_section")
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)

    return final_chunks

def retrieve_node(state: GraphState) -> GraphState:
    timing = {}
    t0 = time.perf_counter()

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

    # 1. Hybrid Search
    t_start = time.perf_counter()
    docs = runtime.ensemble_retriever(expanded_query, k=HYBRID_CANDIDATE_K)
    timing['hybrid_retrieval_ms'] = (time.perf_counter() - t_start) * 1000

    # 2. Rerank
    t_start = time.perf_counter()
    reranked_docs = rerank_documents(question, docs, max_docs=RERANK_TOP_K)
    timing['rerank_ms'] = (time.perf_counter() - t_start) * 1000

    # 3. Parent Expansion
    t_start = time.perf_counter()
    expanded_docs = expand_parent_sections(reranked_docs)
    timing['parent_expansion_ms'] = (time.perf_counter() - t_start) * 1000

    # 4. Contextual Compression
    t_start = time.perf_counter()
    compressed_docs = compress_context(question, expanded_docs)
    timing['compression_ms'] = (time.perf_counter() - t_start) * 1000

    context_blocks = []
    top_pages = set()

    for i, doc in enumerate(compressed_docs):
        metadata = doc.metadata or {}
        page = metadata.get("page", "unknown")
        source = metadata.get("source", runtime.document_name or "document")
        score = metadata.get("hybrid_score", 0)
        rerank_score = metadata.get("rerank_score", 0)
        
        doc_type = metadata.get("type", "text")
        if metadata.get("is_table"):
            doc_type = "table"

        if page != "unknown" and doc_type != "image" and i < IMAGE_TOP_PAGES:
            top_pages.add(page)

        block = {
            "source": str(source),
            "page": str(page),
            "section": str(metadata.get("section", "")),
            "title": str(metadata.get("title", "")),
            "type": doc_type,
            "chunk_type": str(metadata.get("chunk_type", "")),
            "relevance_score": score,
            "rerank_score": rerank_score
        }
        
        if doc_type == "image":
            block["image_base64"] = metadata.get("image_base64")
            block["image_mime"] = metadata.get("image_mime", "image/png")
        else:
            block["content"] = doc.page_content

        context_blocks.append(block)

    # 5. Image Attachment
    image_keys_added = set()
    for page_key in top_pages:
        try:
            page_key = int(page_key)
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
                "rerank_score": 0,
                "section": "",
                "title": "",
                "chunk_type": "image"
            })
            image_keys_added.add(image_key)

            if len(image_keys_added) >= MAX_IMAGES:
                break

        if len(image_keys_added) >= MAX_IMAGES:
            break

    timing['total_retrieval_ms'] = (time.perf_counter() - t0) * 1000

    return {**state, "context": context_blocks, "retrieval_timing": timing}


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

    prompt = f"""You are an intelligent assistant that answers queries using ONLY the provided document context.

---

**SOURCE OF TRUTH

* Use ONLY the given context.
* Do NOT use external knowledge or assumptions.
* If information is missing, respond exactly:
  "I don't have enough information in the document to answer that."

---

**CORE RULES**

* Do NOT hallucinate or fabricate details.
* Preserve original meaning, values, and conditions.
* Do NOT omit important constraints or exceptions.

---

**REASONING & CALCULATIONS (ALLOWED)**

* You are allowed to reason, explain, and perform calculations.
* Use ONLY rules, formulas, and data present in the context.
* Do NOT invent logic or formulas.

---

**IMAGE USAGE RULES**

* Include supporting images ONLY if:
  - They are present in the provided document context, AND
  - They are directly relevant to answering the query.
* Do NOT generate, assume, or describe images that are not explicitly present.

---

**RULE-BASED QUESTIONS**

* If the answer exists directly in the document:
  → Return it AS-IS (no restructuring, no summarization).

* If application is required:
  Structure your answer as:

  1. Rule from Document
  2. Given Input
  3. Application / Calculation
  4. Final Answer

---

**STRUCTURED CONTENT**

* Tables → Reconstruct completely (all rows & columns).
* Lists / Clauses → Preserve numbering and hierarchy.
* Do NOT skip or summarize structured data.

---

**REFERENCES**

* Include section/page references if available.
* Do NOT create fake references.

---

**DECISION FLOW**

* Direct answer → Return as-is
* Explanation → Explain from context
* Rule application → Apply with steps
* Partial info → Answer + mention gap
* No info → Say not available

---

**STYLE**

* Clear, formal, and readable
* Structured > descriptive
* Complete > summarized
* Visual + text combination preferred when available

 

Context:
{context_str}

Question: {state["question"]}   

Answer:"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)

    usage = response.response_metadata.get("token_usage", {})
    current_usage = {"input": 0, "output": 0}
    
    if usage:
        in_toks = usage.get("prompt_tokens", 0)
        out_toks = usage.get("completion_tokens", 0)
    else:
        in_toks = count_tokens(prompt, "gpt-4o-mini")
        out_toks = count_tokens(response.content, "gpt-4o-mini")
        
    runtime.total_llm_input_tokens += in_toks
    runtime.total_llm_output_tokens += out_toks
    current_usage["input"] = in_toks
    current_usage["output"] = out_toks

    return {**state, "answer": response.content, "token_usage": current_usage}


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
        "retrieval_method": "hybrid (FAISS + BM25)",
        "has_faiss": runtime.faiss_index is not None,
        "has_bm25": runtime.bm25_retriever is not None,
        "has_images": len(runtime.page_images) > 0,
        "reranker_type": "lightweight_rule_based",
        "compression_type": "rule_based",
        "loader_type": runtime.loader_type or "unknown"
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

    final_state = rag_graph.invoke({"question": question, "context": [], "answer": "", "retrieval_timing": None, "token_usage": None})
    return AskResponse(
        answer=final_state["answer"],
        chunks=final_state["context"],
        token_usage=final_state.get("token_usage") or {"input": 0, "output": 0},
        retrieval_timing=final_state.get("retrieval_timing")
    )
@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    require_openai_key()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if runtime.faiss_index is None:
        raise HTTPException(status_code=400, detail="Upload and process a document first.")

    async def generate():
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()

        # 1. Run retrieval in a thread (it's synchronous)
        with ThreadPoolExecutor() as pool:
            state = await loop.run_in_executor(
                pool,
                lambda: retrieve_node({
                    "question": question,
                    "context": [],
                    "answer": "",
                    "retrieval_timing": None,
                    "token_usage": None,
                })
            )

        context_blocks = state["context"]
        retrieval_timing = state.get("retrieval_timing")

        # 2. Build context string (mirrors generate_node logic)
        context_parts = []
        for c in context_blocks:
            if c.get("type") == "image":
                context_parts.append(
                    f"[Source: {c['source']}, Page: {c['page']}]\n[IMAGE available for UI rendering]"
                )
            elif c.get("type") == "table":
                context_parts.append(
                    f"[Source: {c['source']}, Page: {c['page']}]\n[TABLE]\n{c['content']}"
                )
            else:
                context_parts.append(
                    f"[Source: {c['source']}, Page: {c['page']}]\n{c['content']}"
                )

        context_str = "\n\n---\n\n".join(context_parts)
        prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, question=question)

        # 3. Stream LLM tokens
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        full_response = ""
        in_toks = count_tokens(prompt, "gpt-4o-mini")

        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # 4. Final done event with metadata
        out_toks = count_tokens(full_response, "gpt-4o-mini")
        runtime.total_llm_input_tokens += in_toks
        runtime.total_llm_output_tokens += out_toks

        yield f"data: {json.dumps({'type': 'done', 'chunks': context_blocks, 'token_usage': {'input': in_toks, 'output': out_toks}, 'retrieval_timing': retrieval_timing})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
