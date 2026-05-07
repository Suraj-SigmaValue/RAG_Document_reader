import os

from dotenv import load_dotenv


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
    convert_from_path = None
    pytesseract = None
