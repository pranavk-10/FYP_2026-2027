"""
Blood Report OCR Pipeline
==========================
Uses Mistral OCR to extract text from blood report images/PDFs,
then passes the raw markdown to Groq LLM for structured JSON extraction.

Supports two input modes:
  1. process_blood_report(file_path) — for local file paths (original Ojas function)
  2. process_blood_report_bytes(file_bytes, filename) — for FastAPI UploadFile bytes
"""

import os
import base64
import json
from mistralai.client import Mistral  # mistralai SDK v3.0.0
from dotenv import load_dotenv

# Package-relative import (works when called from FastAPI or any external module)
from .ocr_structurer import structure_ocr_to_json

load_dotenv()


def _get_mistral_client():
    """Returns authenticated Mistral client."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY not found in environment variables.")
    return Mistral(api_key=api_key)


def _determine_document_type(base64_data: str, ext: str) -> dict:
    """Determines the Mistral OCR document type from file extension."""
    if ext == ".pdf":
        return {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_data}"
        }
    elif ext in [".jpg", ".jpeg"]:
        return {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_data}"
        }
    elif ext == ".png":
        return {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{base64_data}"
        }
    else:
        raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .jpg, .jpeg, .png")


def _run_ocr_and_structure(document: dict) -> dict:
    """
    Core OCR pipeline: sends document to Mistral OCR, gets markdown,
    then passes to Groq for structured JSON extraction.
    """
    client = _get_mistral_client()

    print("Sending to Mistral OCR...")
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document
        )
    except Exception as e:
        print(f"❌ Mistral OCR API Error: {e}")
        return {"error": f"Mistral OCR failed: {str(e)}"}

    # Combine markdown from all pages (in case the blood report is multiple pages)
    full_markdown = "\n".join([page.markdown for page in ocr_response.pages])

    if not full_markdown.strip():
        return {"error": "OCR returned empty text — the image may be unreadable."}

    print(f"OCR extracted {len(full_markdown)} characters of markdown text.")

    # Pass the raw markdown to Groq for structuring into CBCReportJSON
    final_json = structure_ocr_to_json(full_markdown)

    # Add the raw OCR markdown to the output for debugging/transparency
    final_json["_raw_ocr_markdown"] = full_markdown

    return final_json


# =====================================================================
# PUBLIC API: Two entry points (file path OR raw bytes)
# =====================================================================

def process_blood_report(file_path: str) -> dict:
    """
    Original function (Ojas's interface) — processes a blood report from a file path.

    Args:
        file_path: Absolute path to blood report image (.jpg, .png) or PDF (.pdf).

    Returns:
        Structured CBCReportJSON dict with hemoglobin, WBC, platelets, abnormalities.
    """
    print(f"Starting OCR extraction on: {file_path}")

    if not os.path.exists(file_path):
        return {"error": f"File not found: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()

    # Encode file to base64
    print("Encoding file...")
    with open(file_path, "rb") as f:
        base64_data = base64.b64encode(f.read()).decode("utf-8")

    document = _determine_document_type(base64_data, ext)
    return _run_ocr_and_structure(document)


def process_blood_report_bytes(file_bytes: bytes, filename: str) -> dict:
    """
    FastAPI-compatible function — processes a blood report from raw bytes.
    This is what the /api/ocr/blood-report endpoint calls.

    Args:
        file_bytes: Raw file content (from UploadFile.read()).
        filename: Original filename (used to determine file type from extension).

    Returns:
        Structured CBCReportJSON dict with hemoglobin, WBC, platelets, abnormalities.
    """
    print(f"Starting OCR extraction on uploaded file: {filename}")

    ext = os.path.splitext(filename)[1].lower()

    # Encode bytes to base64
    base64_data = base64.b64encode(file_bytes).decode("utf-8")

    document = _determine_document_type(base64_data, ext)
    return _run_ocr_and_structure(document)


if __name__ == "__main__":
    # Point this to your actual blood report PDF/Image
    # TEST_FILE = "/Users/ojaspatil/Desktop/bloodreport.png"
    TEST_FILE = "test_blood_report.png"  # Change to your local test file

    if os.path.exists(TEST_FILE):
        print("\n--- INITIATING PIPELINE ---")
        result = process_blood_report(TEST_FILE)

        print("\n==================================================")
        print("FINAL STRUCTURED JSON OUTPUT:")
        print("==================================================")
        print(json.dumps(result, indent=2))

    else:
        print(f"❌ File not found: {TEST_FILE}")