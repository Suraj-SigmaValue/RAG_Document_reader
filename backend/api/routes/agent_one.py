import json
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from backend.agents.UI_dashboard.main import build_context_string, rag_graph, retrieve_node, token_usage
from backend.agents.UI_dashboard.prompts import RAG_PROMPT_TEMPLATE
from backend.agents.UI_dashboard.tools import (
    create_faiss_retriever,
    create_hybrid_retriever,
    extract_images_from_pdf,
    hybrid_chunking,
    load_documents,
)
from backend.core.config import OPENAI_API_KEY
from backend.core.security import require_openai_key
from backend.database.db import runtime
from backend.schemas import AskRequest, AskResponse
from backend.utils.helpers import count_tokens


router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"ok": True}


@router.get("/")
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


@router.get("/status")
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
        "loader_type": runtime.loader_type or "unknown",
    }


@router.post("/documents")
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

        chunks = hybrid_chunking(documents)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document did not produce any text chunks.")

        create_faiss_retriever(chunks)
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


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    require_openai_key()

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if runtime.faiss_index is None:
        raise HTTPException(status_code=400, detail="Upload and process a document first.")

    final_state = rag_graph.invoke(
        {"question": question, "context": [], "answer": "", "retrieval_timing": None, "token_usage": None}
    )
    return AskResponse(
        answer=final_state["answer"],
        chunks=final_state["context"],
        token_usage=final_state.get("token_usage") or {"input": 0, "output": 0},
        retrieval_timing=final_state.get("retrieval_timing"),
    )


@router.post("/ask/stream")
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

        with ThreadPoolExecutor() as pool:
            state = await loop.run_in_executor(
                pool,
                lambda: retrieve_node(
                    {
                        "question": question,
                        "context": [],
                        "answer": "",
                        "retrieval_timing": None,
                        "token_usage": None,
                    }
                ),
            )

        context_blocks = state["context"]
        retrieval_timing = state.get("retrieval_timing")
        context_str = build_context_string(context_blocks)
        prompt = RAG_PROMPT_TEMPLATE.format(context_str=context_str, question=question)

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            api_key=OPENAI_API_KEY,
            streaming=True,
        )

        full_response = ""
        in_tokens = count_tokens(prompt, "gpt-4o-mini")

        async for chunk in llm.astream(prompt):
            token = chunk.content
            if token:
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        out_tokens = count_tokens(full_response, "gpt-4o-mini")
        runtime.total_llm_input_tokens += in_tokens
        runtime.total_llm_output_tokens += out_tokens

        yield f"data: {json.dumps({'type': 'done', 'chunks': context_blocks, 'token_usage': {'input': in_tokens, 'output': out_tokens}, 'retrieval_timing': retrieval_timing})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
