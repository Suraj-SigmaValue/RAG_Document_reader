import re
from typing import List

import faiss
import numpy as np
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import (
    HYBRID_CANDIDATE_K,
    MAX_CONTEXT_CHARS,
    OPENAI_API_KEY,
    PARENT_EXPAND_MAX_EXTRA,
    PARENT_EXPAND_TOP_K,
    RERANK_TOP_K,
    RETRIEVAL_BM25_K,
    RETRIEVAL_FAISS_K,
)
from backend.database.db import runtime
from backend.utils.helpers import is_table_like, is_toc_like, preprocess_for_bm25


def create_faiss_retriever(chunks: List[Document]):
    runtime.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=OPENAI_API_KEY)
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = runtime.embeddings.embed_documents(chunk_texts)
    runtime.chunks = chunks

    embeddings_array = np.array(chunk_embeddings).astype("float32")
    dimension = len(embeddings_array[0])
    index = faiss.IndexFlatIP(dimension)

    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)

    runtime.faiss_index = index
    return index


def create_hybrid_retriever(chunks: List[Document], vector_weight: float = 0.6, bm25_weight: float = 0.4):
    bm25_retriever = BM25Retriever.from_documents(chunks, preprocess_func=preprocess_for_bm25)
    bm25_retriever.k = RETRIEVAL_BM25_K
    runtime.bm25_retriever = bm25_retriever

    def hybrid_search(query: str, k: int = HYBRID_CANDIDATE_K) -> List[Document]:
        bm25_docs = bm25_retriever.invoke(query)

        query_embedding = runtime.embeddings.embed_query(query)
        query_array = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_array)

        distances, indices = runtime.faiss_index.search(query_array, RETRIEVAL_FAISS_K)

        faiss_docs = []
        for index, distance in zip(indices[0], distances[0]):
            if index != -1 and index < len(runtime.chunks):
                doc = runtime.chunks[index]
                doc.metadata["faiss_score"] = float(distance)
                doc.metadata["faiss_rank"] = len(faiss_docs) + 1
                faiss_docs.append(doc)

        scores = {}

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.page_content[:100]
            scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (rank + 60)
            doc.metadata["bm25_rank"] = rank + 1

        for rank, doc in enumerate(faiss_docs):
            doc_id = doc.page_content[:100]
            scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (rank + 60)

        all_docs = bm25_docs + faiss_docs
        unique_docs = {}
        for doc in all_docs:
            doc_id = doc.page_content[:100]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = doc
                unique_docs[doc_id].metadata["hybrid_score"] = scores[doc_id]

        sorted_docs = sorted(unique_docs.values(), key=lambda item: item.metadata.get("hybrid_score", 0), reverse=True)
        return sorted_docs[:k]

    return hybrid_search


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
    q_words = set(re.findall(r"\w+", q_lower))

    table_keywords = {"table", "rate", "area", "cost", "fsi", "calculation", "statement", "schedule"}
    has_table_intent = any(keyword in q_words for keyword in table_keywords)

    for doc in docs:
        score = doc.metadata.get("hybrid_score", 0)
        content_lower = doc.page_content.lower()
        title_lower = str(doc.metadata.get("title", "")).lower()
        section_lower = str(doc.metadata.get("section", "")).lower()

        content_words = set(re.findall(r"\w+", content_lower))
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

    reranked = sorted(docs, key=lambda item: item.metadata.get("rerank_score", 0), reverse=True)
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
        if section:
            target_sections.add(section)
        if parent_section:
            target_sections.add(parent_section)

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
    q_words = set(re.findall(r"\w+", question.lower()))

    total_chars = 0

    for doc in docs:
        if total_chars >= MAX_CONTEXT_CHARS:
            break

        if doc.metadata.get("is_table") or doc.metadata.get("type") == "image":
            compressed_content = doc.page_content
        else:
            paragraphs = doc.page_content.split("\n\n")
            kept_paragraphs = []
            for paragraph in paragraphs:
                paragraph_words = set(re.findall(r"\w+", paragraph.lower()))
                if len(q_words & paragraph_words) > 0 or len(paragraph) < 50:
                    kept_paragraphs.append(paragraph)

            compressed_content = "\n\n".join(kept_paragraphs)
            if len(compressed_content) < len(doc.page_content) * 0.2:
                compressed_content = doc.page_content

        new_doc = Document(page_content=compressed_content, metadata=doc.metadata.copy())
        compressed_docs.append(new_doc)
        total_chars += len(compressed_content)

    return compressed_docs


def structure_based_split(documents):
    structured_chunks = []
    section_pattern = r"\n(?=\d+\.\d+(?:\.\d+)*\s+[A-Z][A-Za-z ]{4,80})"

    for doc in documents:
        if doc.metadata.get("type") == "image":
            structured_chunks.append(doc)
            continue

        text = doc.page_content
        page = doc.metadata.get("page", None)
        sections = re.split(section_pattern, text)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            match = re.match(r"(\d+(\.\d+)*)\s*(.*)", section)
            section_id = match.group(1) if match else None
            title = match.group(3)[:100] if match else None

            parent_section = None
            if section_id and "." in section_id:
                parts = section_id.split(".")
                parent_section = ".".join(parts[:-1])

            structured_chunks.append(
                Document(
                    page_content=section,
                    metadata={
                        "page": page,
                        "source": doc.metadata.get("source", runtime.document_name or "document"),
                        "section": section_id,
                        "parent_section": parent_section,
                        "title": title,
                        "chunk_type": "section",
                        "is_table": is_table_like(section),
                    },
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
            for index, sub_chunk in enumerate(sub_chunks):
                sub_chunk.metadata.update(doc.metadata)
                sub_chunk.metadata["chunk_type"] = "sub_chunk"
                sub_chunk.metadata["sub_chunk_index"] = index
                sub_chunk.metadata["parent_section"] = doc.metadata.get("parent_section")
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(doc)

    return final_chunks
