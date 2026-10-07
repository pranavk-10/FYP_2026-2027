import os
import io
import json
import tempfile
from typing import Optional, Dict, Any
import pytesseract
from PIL import Image

# Automatically configure Tesseract binary path on Windows if present
if os.name == 'nt':
    standard_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(standard_tesseract):
        pytesseract.pytesseract.tesseract_cmd = standard_tesseract

# Support relative or absolute import of ocr_parser
try:
    from .ocr_parser import extract_cbc_json
except ImportError:
    try:
        from ocr_pipeline.ocr_parser import extract_cbc_json
    except ImportError:
        from ocr_parser import extract_cbc_json


def extract_text(file_path: str) -> str:
    """Extracts raw text from an image or PDF file on disk."""
    try:
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        extension = os.path.splitext(file_path)[1].lower()

        # IMAGE
        if extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
            img = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(img)
            return extracted_text.strip()

        # PDF
        elif extension == ".pdf":
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(file_path, dpi=300)
                all_text = []
                for page_number, page in enumerate(pages, start=1):
                    text = pytesseract.image_to_string(page)
                    all_text.append(f"--- Page {page_number} ---\n{text.strip()}")
                return "\n\n".join(all_text)
            except Exception as pdf_err:
                # Graceful fallback: try pypdf if poppler is missing
                try:
                    import pypdf
                    reader = pypdf.PdfReader(file_path)
                    all_text = [p.extract_text() or "" for p in reader.pages]
                    return "\n\n".join(all_text).strip()
                except Exception:
                    raise pdf_err

        else:
            return f"Unsupported file type: {extension}"

    except Exception as e:
        return f"An error occurred: {e}"


def extract_text_from_bytes(file_bytes: bytes, filename: str = "report.jpg") -> str:
    """Extracts raw text directly from in-memory file bytes (image or PDF)."""
    extension = os.path.splitext(filename)[1].lower()
    if not extension:
        extension = ".jpg"

    # Images: Process directly in-memory with PIL & pytesseract
    if extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        print(f"[OCR] Image format detected ({extension}). Running Tesseract OCR...")
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img).strip()
        print(f"[OCR] Tesseract extracted {len(text)} characters.")
        return text

    # PDFs: Try pdf2image via temp file, with pypdf fallback
    elif extension == ".pdf":
        print(f"[OCR] PDF document detected. Converting pages to 300 DPI images...")
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(file_bytes, dpi=300)
            print(f"[OCR] Found {len(pages)} page(s). Running OCR per page...")
            all_text = []
            for page_number, page in enumerate(pages, start=1):
                text = pytesseract.image_to_string(page).strip()
                print(f"[OCR]   -> Page {page_number}/{len(pages)}: {len(text)} characters extracted")
                all_text.append(f"--- Page {page_number} ---\n{text}")
            return "\n\n".join(all_text)
        except Exception as e:
            print(f"[OCR] pdf2image conversion failed ({str(e)}). Falling back to pypdf...")
            try:
                import pypdf
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                all_text = [p.extract_text() or "" for p in reader.pages]
                text = "\n\n".join(all_text).strip()
                print(f"[OCR] pypdf extracted {len(text)} characters across {len(reader.pages)} page(s).")
                return text
            except Exception as err:
                raise RuntimeError(f"Failed to extract text from PDF: {str(err)}")

    else:
        raise ValueError(f"Unsupported file format: {extension}")


def _enrich_clinical_flags(raw_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enriches raw CBC JSON with clinical flags (low/normal/high)
    compatible with downstream Multi-Agent LangGraph debate and frontend display.
    """
    cbc = raw_json.get("cbc", {})
    abnormalities = []

    # Hemoglobin (Ref: ~12.0 - 17.5 g/dL)
    hb_data = cbc.get("hemoglobin") or {}
    hb_val = hb_data.get("value")
    hb_unit = hb_data.get("unit") or "g/dL"
    hb_flag = "normal"
    if hb_val is not None:
        if hb_val < 12.0:
            hb_flag = "low"
            abnormalities.append(f"Low Hemoglobin ({hb_val} {hb_unit}) — anemia / blood loss")
        elif hb_val > 17.5:
            hb_flag = "high"
            abnormalities.append(f"Elevated Hemoglobin ({hb_val} {hb_unit}) — polycythemia")

    # WBC (Ref: ~4,000 - 11,000 /µL)
    wbc_data = cbc.get("wbc") or {}
    wbc_val = wbc_data.get("value")
    wbc_unit = wbc_data.get("unit") or "/µL"
    wbc_flag = "normal"
    if wbc_val is not None:
        if wbc_val < 4000:
            wbc_flag = "low"
            abnormalities.append(f"Low WBC ({wbc_val} {wbc_unit}) — leukopenia")
        elif wbc_val > 11000:
            wbc_flag = "high"
            abnormalities.append(f"Elevated WBC ({wbc_val} {wbc_unit}) — leukocytosis / inflammation")

    # Platelets (Ref: ~150,000 - 450,000 /µL)
    plt_data = cbc.get("platelets") or {}
    plt_val = plt_data.get("value")
    plt_unit = plt_data.get("unit") or "/µL"
    plt_flag = "normal"
    if plt_val is not None:
        if plt_val < 150000:
            plt_flag = "low"
            abnormalities.append(f"Low Platelets ({plt_val} {plt_unit}) — thrombocytopenia")
        elif plt_val > 450000:
            plt_flag = "high"
            abnormalities.append(f"Elevated Platelets ({plt_val} {plt_unit}) — thrombocytosis")

    return {
        "patient": raw_json.get("patient", {}),
        "cbc": cbc,
        "hemoglobin": {"value": hb_val, "unit": hb_unit, "flag": hb_flag} if hb_val is not None else None,
        "wbc_count": {"value": wbc_val, "unit": wbc_unit, "flag": wbc_flag} if wbc_val is not None else None,
        "platelets": {"value": plt_val, "unit": plt_unit, "flag": plt_flag} if plt_val is not None else None,
        "other_abnormalities": abnormalities
    }


def process_blood_report_bytes(file_bytes: bytes, filename: str = "report.jpg") -> Dict[str, Any]:
    """End-to-end processing from bytes: OCR extraction -> Groq JSON structuring -> clinical flags."""
    print(f"\n==================== [OCR PIPELINE] ====================")
    print(f"[OCR] Received document: '{filename}' ({len(file_bytes)/1024:.1f} KB)")
    try:
        print(f"[OCR] Step 1/3: Extracting text via Tesseract OCR...")
        raw_text = extract_text_from_bytes(file_bytes, filename)
        if not raw_text or len(raw_text.strip()) < 5:
            print(f"[OCR] ❌ Error: No readable text detected in document.")
            print(f"=========================================================\n")
            return {"error": "OCR failed to detect any text in the uploaded document."}

        print(f"[OCR] Step 2/3: Structuring CBC values via Groq LLM (openai/gpt-oss-20b)...")
        raw_json = extract_cbc_json(raw_text)

        print(f"[OCR] Step 3/3: Evaluating clinical reference flags...")
        enriched = _enrich_clinical_flags(raw_json)
        enriched["raw_ocr_text_preview"] = raw_text[:300] + ("..." if len(raw_text) > 300 else "")

        pat = enriched.get("patient", {})
        pat_str = f"Name: {pat.get('name') or 'N/A'}, Age: {pat.get('age') or 'N/A'}, Gender: {pat.get('gender') or 'N/A'}"
        print(f"[OCR]   - Patient: {pat_str}")

        hb = enriched.get("hemoglobin")
        wbc = enriched.get("wbc_count")
        plt = enriched.get("platelets")
        hb_str = f"{hb['value']} {hb['unit']} [{hb['flag'].upper()}]" if hb and hb.get("value") is not None else "N/A"
        wbc_str = f"{wbc['value']} {wbc['unit']} [{wbc['flag'].upper()}]" if wbc and wbc.get("value") is not None else "N/A"
        plt_str = f"{plt['value']} {plt['unit']} [{plt['flag'].upper()}]" if plt and plt.get("value") is not None else "N/A"
        print(f"[OCR]   - Key Labs: Hb: {hb_str} | WBC: {wbc_str} | Platelets: {plt_str}")

        if enriched.get("other_abnormalities"):
            print(f"[OCR]   - Abnormalities: {'; '.join(enriched['other_abnormalities'])}")
        else:
            print(f"[OCR]   - Clinical Flags: All key parameters within normal reference ranges.")

        print(f"[OCR] [SUCCESS] CBC Report Processing Complete.")
        print(f"=========================================================\n")
        return enriched
    except Exception as e:
        print(f"[OCR] [ERROR] Pipeline Exception: {str(e)}")
        print(f"=========================================================\n")
        return {"error": f"Blood Report OCR Pipeline Error: {str(e)}"}


def process_blood_report(file_path: str) -> Dict[str, Any]:
    """End-to-end processing from a disk file path."""
    with open(file_path, "rb") as f:
        return process_blood_report_bytes(f.read(), os.path.basename(file_path))


# ==========================================
# CLI / STANDALONE TEST
# ==========================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Testing OCR on: {test_file}")
        res = process_blood_report(test_file)
        print("\n=== EXTRACTED CBC RESULT ===")
        print(json.dumps(res, indent=4))
    else:
        print("OCR Pipeline Module ready. Pass a file path to test.")