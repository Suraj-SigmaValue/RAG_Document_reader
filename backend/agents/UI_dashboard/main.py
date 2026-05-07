import time

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from backend.agents.UI_dashboard.prompts import RAG_PROMPT_TEMPLATE
from backend.core.config import (
    HYBRID_CANDIDATE_K,
    IMAGE_TOP_PAGES,
    MAX_IMAGES,
    OPENAI_API_KEY,
    RERANK_TOP_K,
)
from backend.database.db import runtime
from backend.schemas import GraphState
from backend.tools.search import compress_context, expand_parent_sections, rerank_documents
from backend.utils.helpers import count_tokens


def retrieve_node(state: GraphState) -> GraphState:
    timing = {}
    start_time = time.perf_counter()

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

    step_start = time.perf_counter()
    docs = runtime.ensemble_retriever(expanded_query, k=HYBRID_CANDIDATE_K)
    timing["hybrid_retrieval_ms"] = (time.perf_counter() - step_start) * 1000

    step_start = time.perf_counter()
    reranked_docs = rerank_documents(question, docs, max_docs=RERANK_TOP_K)
    timing["rerank_ms"] = (time.perf_counter() - step_start) * 1000

    step_start = time.perf_counter()
    expanded_docs = expand_parent_sections(reranked_docs)
    timing["parent_expansion_ms"] = (time.perf_counter() - step_start) * 1000

    step_start = time.perf_counter()
    compressed_docs = compress_context(question, expanded_docs)
    timing["compression_ms"] = (time.perf_counter() - step_start) * 1000

    context_blocks = []
    top_pages = set()

    for index, doc in enumerate(compressed_docs):
        metadata = doc.metadata or {}
        page = metadata.get("page", "unknown")
        source = metadata.get("source", runtime.document_name or "document")
        score = metadata.get("hybrid_score", 0)
        rerank_score = metadata.get("rerank_score", 0)

        doc_type = metadata.get("type", "text")
        if metadata.get("is_table"):
            doc_type = "table"

        if page != "unknown" and doc_type != "image" and index < IMAGE_TOP_PAGES:
            top_pages.add(page)

        block = {
            "source": str(source),
            "page": str(page),
            "section": str(metadata.get("section", "")),
            "title": str(metadata.get("title", "")),
            "type": doc_type,
            "chunk_type": str(metadata.get("chunk_type", "")),
            "relevance_score": score,
            "rerank_score": rerank_score,
        }

        if doc_type == "image":
            block["image_base64"] = metadata.get("image_base64")
            block["image_mime"] = metadata.get("image_mime", "image/png")
        else:
            block["content"] = doc.page_content

        context_blocks.append(block)

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

            context_blocks.append(
                {
                    "source": str(metadata.get("source", runtime.document_name or "document")),
                    "page": str(page_key),
                    "type": "image",
                    "image_base64": metadata.get("image_base64"),
                    "image_mime": metadata.get("image_mime", "image/png"),
                    "relevance_score": metadata.get("hybrid_score", 0),
                    "rerank_score": 0,
                    "section": "",
                    "title": "",
                    "chunk_type": "image",
                }
            )
            image_keys_added.add(image_key)

            if len(image_keys_added) >= MAX_IMAGES:
                break

        if len(image_keys_added) >= MAX_IMAGES:
            break

    timing["total_retrieval_ms"] = (time.perf_counter() - start_time) * 1000
    return {**state, "context": context_blocks, "retrieval_timing": timing}


def build_context_string(context_blocks):
    context_parts = []

    for context in context_blocks:
        if context.get("type") == "image":
            context_parts.append(
                f"[Source: {context['source']}, Page: {context['page']}]\n"
                "[IMAGE available for UI rendering]"
            )
        elif context.get("type") == "table":
            context_parts.append(
                f"[Source: {context['source']}, Page: {context['page']}]\n"
                f"[TABLE]\n{context['content']}"
            )
        else:
            context_parts.append(
                f"[Source: {context['source']}, Page: {context['page']}]\n"
                f"{context['content']}"
            )

    return "\n\n---\n\n".join(context_parts)


def generate_node(state: GraphState) -> GraphState:
    if not state["context"]:
        return {**state, "answer": "No relevant content found in the document."}

    context_str = build_context_string(state["context"])
    prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, question=state["question"])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=OPENAI_API_KEY)
    response = llm.invoke(prompt)

    usage = response.response_metadata.get("token_usage", {})
    current_usage = {"input": 0, "output": 0}

    if usage:
        in_tokens = usage.get("prompt_tokens", 0)
        out_tokens = usage.get("completion_tokens", 0)
    else:
        in_tokens = count_tokens(prompt, "gpt-4o-mini")
        out_tokens = count_tokens(response.content, "gpt-4o-mini")

    runtime.total_llm_input_tokens += in_tokens
    runtime.total_llm_output_tokens += out_tokens
    current_usage["input"] = in_tokens
    current_usage["output"] = out_tokens

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
