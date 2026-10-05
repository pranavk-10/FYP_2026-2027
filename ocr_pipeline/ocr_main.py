import os
import json
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from ocr_parser import extract_cbc_json


def extract_text(file_path):
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        # Get file extension
        extension = os.path.splitext(file_path)[1].lower()

        # ==========================================
        # IMAGE
        # ==========================================
        if extension in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:

            img = Image.open(file_path)

            extracted_text = pytesseract.image_to_string(img)

            return extracted_text.strip()

        # ==========================================
        # PDF
        # ==========================================
        elif extension == ".pdf":

            # Convert PDF pages into images
            pages = convert_from_path(
                file_path,
                dpi=300
            )

            all_text = []

            # OCR each page
            for page_number, page in enumerate(pages, start=1):

                text = pytesseract.image_to_string(page)

                all_text.append(
                    f"--- Page {page_number} ---\n{text.strip()}"
                )

            return "\n\n".join(all_text)

        # ==========================================
        # UNSUPPORTED FILE
        # ==========================================
        else:
            return f"Unsupported file type: {extension}"

    except Exception as e:
        return f"An error occurred: {e}"


# ==========================================
# TEST
# ==========================================

file_path = "/Users/ojaspatil/Desktop/tst.jpg"

# STEP 1: OCR

extracted_text = extract_text(file_path)

print("\n==============================")

print("       EXTRACTED TEXT")

print("==============================\n")

print(extracted_text)

# STEP 2: Send OCR text to Groq

print("\n==============================")

print("       GROQ JSON")

print("==============================\n")

cbc_data = extract_cbc_json(extracted_text)

# STEP 3: Pretty print JSON

print(

    json.dumps(

        cbc_data,

        indent=4

    )

)