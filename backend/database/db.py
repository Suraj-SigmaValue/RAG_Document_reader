from typing import Optional


class RagRuntime:
    def __init__(self) -> None:
        self.faiss_index = None
        self.chunks = []
        self.page_images = {}
        self.bm25_retriever = None
        self.document_name: Optional[str] = None
        self.chunk_count = 0
        self.total_llm_input_tokens = 0
        self.total_llm_output_tokens = 0
        self.embeddings = None
        self.loader_type: Optional[str] = None
        self.ensemble_retriever = None


runtime = RagRuntime()
