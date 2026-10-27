"""File parser for text, markdown, PDF, and DOCX documents."""

from __future__ import annotations

import io
from typing import BinaryIO, Union


def extract_text_from_file(file_obj: Union[BinaryIO, io.BytesIO, bytes], filename: str) -> str:
    """Extract clean text from uploaded files (txt, md, pdf, docx)."""
    filename_lower = filename.lower()
    
    if isinstance(file_obj, bytes):
        file_obj = io.BytesIO(file_obj)

    if filename_lower.endswith((".txt", ".md", ".csv")):
        content = file_obj.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)

    elif filename_lower.endswith(".pdf"):
        try:
            import pypdf

            reader = pypdf.PdfReader(file_obj)
            extracted = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    extracted.append(text)
            return "\n\n".join(extracted).strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF file '{filename}': {e}") from e

    elif filename_lower.endswith(".docx"):
        try:
            import docx

            doc = docx.Document(file_obj)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs).strip()
        except Exception as e:
            raise ValueError(f"Error reading Word document '{filename}': {e}") from e

    else:
        # Fallback to UTF-8 text decode
        content = file_obj.read()
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="ignore")
        return str(content)
