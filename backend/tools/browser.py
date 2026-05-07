import base64
import json
import os
from typing import List

import fitz
from fastapi import HTTPException
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

from backend.core.config import OCR_AVAILABLE, POPPLER_PATH, convert_from_path, pytesseract
from backend.database.db import runtime
from backend.utils.helpers import is_table_like


def extract_images_from_pdf(file_path):
    doc = fitz.open(file_path)
    image_docs = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for image_index, image in enumerate(images):
            xref = image[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            encoded = base64.b64encode(image_bytes).decode("utf-8")

            image_docs.append(
                Document(
                    page_content="[IMAGE]",
                    metadata={
                        "page": page_index + 1,
                        "image_base64": encoded,
                        "image_mime": f"image/{image_ext}",
                        "image_index": image_index,
                        "type": "image",
                    },
                )
            )

    return image_docs


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

        docs.append(
            Document(
                page_content=md_content,
                metadata={
                    "page": page_num,
                    "source": "opendataloader",
                    "type": "text",
                    "section": item.get("section_id") or item.get("section"),
                    "title": item.get("section_title") or item.get("title"),
                    "bbox": item.get("bbox") or item.get("bounding_box"),
                    "is_table": is_table_like(md_content),
                },
            )
        )

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

        except Exception as exc:
            print(f"PyPDFLoader failed: {exc}")

        try:
            docs = load_pdf_with_opendataloader(file_path)
            total_text = " ".join(doc.page_content for doc in docs)
            if len(total_text.strip()) > 500:
                runtime.loader_type = "opendataloader"
                return docs

        except Exception as exc:
            print(f"OpenDataLoader failed, skipped safely: {exc}")

        runtime.loader_type = "ocr"
        return load_pdf_with_ocr(file_path)

    if extension == ".docx":
        runtime.loader_type = "docx"
        return Docx2txtLoader(file_path).load()

    raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")
