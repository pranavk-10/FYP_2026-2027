import os
import base64
import json
from mistralai.client import Mistral # Updated import to match current SDK
from dotenv import load_dotenv

# Import the structuring function from your other file
from ocr_structurer import structure_ocr_to_json

load_dotenv()

def encode_file(file_path: str) -> str:
    """Encode a local file into base64."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")

def process_blood_report(file_path: str) -> dict:
    print(f"Starting OCR extraction on: {file_path}")

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not found")
        return {"error": "Missing API key"}

    client = Mistral(api_key=api_key)
    ext = os.path.splitext(file_path)[1].lower()

    # Encode file
    print("Encoding file...")
    base64_file = encode_file(file_path)

    # Determine document type
    if ext == ".pdf":
        document = {
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_file}"
        }
    elif ext in [".jpg", ".jpeg"]:
        document = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{base64_file}"
        }
    elif ext == ".png":
        document = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{base64_file}"
        }
    else:
        print(f"❌ Unsupported file type: {ext}")
        return {"error": f"Unsupported file type: {ext}"}

    # Call OCR
    print("Sending to Mistral OCR...")
    try:
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document=document
        )
    except Exception as e:
        print(f"❌ API Error: {e}")
        return {"error": str(e)}

    # Combine markdown from all pages (in case the blood report is multiple pages)
    full_markdown = "\n".join([page.markdown for page in ocr_response.pages])
    
    # Pass the raw markdown to Groq for structuring
    final_json = structure_ocr_to_json(full_markdown)
    
    return final_json

if __name__ == "__main__":
    # Point this to your actual blood report PDF/Image
    TEST_FILE = "/Users/ojaspatil/Desktop/bloodreport.png"

    if os.path.exists(TEST_FILE):
        print("\n--- INITIATING PIPELINE ---")
        result = process_blood_report(TEST_FILE)
        
        print("\n==================================================")
        print("FINAL STRUCTURED JSON OUTPUT:")
        print("==================================================")
        print(json.dumps(result, indent=2))
        
    else:
        print(f"❌ File not found: {TEST_FILE}")