"""
Blood Report OCR + Structuring Pipeline Module
Extracts text via Tesseract OCR / PDF parsers and structures Complete Blood Count (CBC) data via Groq LLM.
"""

from .ocr_main import (
    extract_text,
    extract_text_from_bytes,
    process_blood_report,
    process_blood_report_bytes,
)
from .ocr_parser import extract_cbc_json

__all__ = [
    "extract_text",
    "extract_text_from_bytes",
    "process_blood_report",
    "process_blood_report_bytes",
    "extract_cbc_json",
]
