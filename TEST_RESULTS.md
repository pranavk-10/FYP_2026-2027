# Comprehensive System Test Results

**Date**: 2026-10-08  
**API Base URL**: `http://127.0.0.1:8000`  
**Framework**: FastAPI + PyTorch (ResNet18) + TensorFlow (DenseNet121) + LangGraph MDT Debate (Groq GPT-OSS) + Tesseract OCR  

---

## Executive Summary

| Test Suite | Modality / Component | Test Input | HTTP Status | Outcome / Verdict |
| :--- | :--- | :--- | :---: | :--- |
| **Test 1** | **Blood Report OCR & CBC Structuring** | `cbc-report-format.pdf` | `200 OK` | Extracted demographics, 13 CBC parameters, normalized units, and clinical reference flags. |
| **Test 2** | **Skin Lesion Specialist + MDT Debate** | `testing/skin.jpeg` | `200 OK` | ResNet18 detected **Melanocytic Nevus (93.5%)**; MDT consensus: **Benign melanocytic nevus (0.93)**. |
| **Test 3** | **ECG Specialist Pipeline** | `testing/ecg.jpg` | `200 OK` | ResNet18 detected **Myocardial Infarction (100.0%)** with ST-segment elevation. |
| **Test 4** | **Chest X-Ray Specialist Pipeline** | `testing/xray.jpeg` | `200 OK` | DenseNet121 detected **No Finding (Normal)** across 20 multi-label pathologies. |
| **Test 5** | **Joint Multimodal Fusion (All 3 Concurrent)** | `ecg.jpg` + `xray.jpeg` + `skin.jpeg` | `200 OK` | All 3 CNNs executed, unified into LangGraph MDT debate; 2 rounds deliberated; consensus: **Refer to Specialist (0.62)**. |

---

## Test 1: Blood Report OCR & CBC Structuring

* **Endpoint**: `POST /api/ocr/blood-report`
* **Test File**: `cbc-report-format.pdf`
* **Technologies**: `pdf2image` + `pytesseract` + Groq LLM (`openai/gpt-oss-20b`) with native JSON Schema

### API Response
```json
{
  "status": "success",
  "filename": "cbc-report-format.pdf",
  "lab_values": {
    "patient": {
      "name": "Mr. Saubhik Bhaumik",
      "age": 27,
      "gender": "M"
    },
    "cbc": {
      "hemoglobin": { "value": 15, "unit": "g/dL" },
      "rbc": { "value": 5, "unit": "million/µL" },
      "wbc": { "value": 5100, "unit": "/µL" },
      "platelets": { "value": 350000, "unit": "/µL" },
      "hematocrit": { "value": 42, "unit": "%" },
      "mcv": { "value": 84, "unit": "fL" },
      "mch": { "value": 30, "unit": "pg" },
      "mchc": { "value": 35.7, "unit": "%" },
      "neutrophils": { "value": 79, "unit": "%" },
      "lymphocytes": { "value": 18, "unit": "%" },
      "monocytes": { "value": 1, "unit": "%" },
      "eosinophils": { "value": 1, "unit": "%" },
      "basophils": { "value": 1, "unit": "%" }
    },
    "hemoglobin": { "value": 15, "unit": "g/dL", "flag": "normal" },
    "wbc_count": { "value": 5100, "unit": "/µL", "flag": "normal" },
    "platelets": { "value": 350000, "unit": "/µL", "flag": "normal" },
    "other_abnormalities": []
  }
}
```

### Key Validations
* ✅ **Unit Normalization**: `5,100 cumm` was normalized to `5100 /µL`.
* ✅ **Demographics Extraction**: Captured patient name, age, and gender accurately.
* ✅ **Clinical Grading**: Flagged Hemoglobin, WBC, and Platelets as `normal`.

---

## Test 2: Skin Lesion Specialist Pipeline & MDT Debate

* **Endpoint**: `POST /api/analyze/skin`
* **Test File**: `testing/skin.jpeg`
* **Model**: PyTorch ResNet18 trained on HAM10000 (7 classes)
* **Debate Engine**: 4-node LangGraph (Advocate $\rightarrow$ Skeptic $\rightarrow$ Evidence $\rightarrow$ Moderator)

### CNN Vision Prediction
* **Primary Diagnosis**: Melanocytic Nevus (`nv`)
* **Probability**: `93.47%`
* **Top Findings**:
  1. Melanocytic Nevus: `93.47%`
  2. Benign Keratosis-like Lesion: `2.09%`
  3. Dermatofibroma: `2.04%`
  4. Vascular Lesion: `1.03%`
  5. Basal Cell Carcinoma: `0.72%`
  6. Melanoma: `0.65%`
* **Evidence Extracted**: Symmetrical pigment network with uniform regular reticular pattern. Absence of atypical streaks or blue-white veil.

### Moderator Verdict
```json
{
  "final_verdict": "Benign melanocytic nevus",
  "confidence_score": 0.93,
  "action": "finalize",
  "audit_trail": "The dermoscopic assessment shows a symmetric lesion with uniform pigment distribution, a regular reticular network, and evenly spaced peripheral globules—classic high-confidence features of a benign melanocytic nevus. The algorithm assigns a 93.5% probability, exceeding the >0.70 threshold for strong positive evidence. No atypical streaks, blue-white veil, asymmetry, regression structures, or vascular patterns suggestive of melanoma or pigmented BCC are present. The skeptic's concerns (early melanoma, dysplastic nevus, pigmented BCC) are theoretical and lack supporting clinical data (no documented growth, symptom change, high-risk location, or family history). Observation with routine surveillance is validated."
}
```

### Key Validations
* ✅ **Single-Round Consensus**: High probability (>70%) with corroborated dermoscopic criteria triggered clean consensus finalization.
* ✅ **No Token Truncation**: Output completed fully with detailed clinical reasoning.

---

## Test 3: ECG Specialist Pipeline

* **Endpoint**: `POST /api/predict/ecg`
* **Test File**: `testing/ecg.jpg`
* **Model**: PyTorch ResNet18 (Cardiology image model)

### CNN Prediction
* **Primary Diagnosis**: Myocardial Infarction (`MI`)
* **Probability**: `100.0%`
* **Calibrated Probability**: `0.999`
* **Evidence**: Marked ST-segment elevation across anatomical leads, pathological Q-waves, indicative of acute coronary syndrome.

---

## Test 4: Chest X-Ray Specialist Pipeline

* **Endpoint**: `POST /api/predict/xray`
* **Test File**: `testing/xray.jpeg`
* **Model**: TensorFlow DenseNet121 (NIH ChestX-ray14 multi-label)

### CNN Prediction
* **Primary Finding**: No Finding (Normal Chest Radiograph)
* **Secondary Signals Evaluated**:
  * Cardiomegaly: Below threshold
  * Pneumothorax: Negative
  * Pleural Effusion: Negative
  * Infiltration / Consolidation: Negative
* **Clinical Significance**: Rules out gross thoracic emergencies (e.g., tension pneumothorax, wide mediastinum).

---

## Test 5: Joint Multimodal Fusion (All 3 Uploaded Simultaneously)

* **Endpoint**: `POST /api/analyze/multimodal`
* **Uploaded Files**:
  * `ecg_file`: `testing/ecg.jpg`
  * `xray_file`: `testing/xray.jpeg`
  * `skin_file`: `testing/skin.jpeg`
* **Execution**: Single HTTP request running 3 concurrent CNN pipelines followed by multi-turn LangGraph deliberation.

### Joint Evidence Packet Synthesized
* `[ECG]` Acute Myocardial Infarction (100% confidence, ST elevation).
* `[XRAY]` No Finding (predominantly clear thoracic study).
* `[DERMOSCOPY]` Melanocytic Nevus (93.5% confidence, uniform regular network).

### Multi-Agent Debate Deliberation (2 Rounds, 6 Turns)

#### Round 1:
* **Advocate**: Prioritized the life-threatening ECG ST-elevation as the emergent diagnostic focus, noting the clear chest X-ray rules out acute aortic dissection and tension pneumothorax, while observing the dermoscopy lesion is an incidental benign nevus.
* **Skeptic**: Challenged the acute MI diagnosis by raising acute pericarditis or repolarization variants, noting the absence of cardiomegaly and lung congestion on X-ray, and questioned whether systemic collagen vascular disease connects the skin lesion and cardiac findings.
* **Evidence-Checker**: Queried WHO HEARTS CVD guidelines in Pinecone; identified that standard criteria require **both** electrophysiological changes and a characteristic rise/fall in cardiac biomarkers (Troponin).
* **Moderator**: Noted disagreement on whether the acute presentation is definitively ischemic without biomarker confirmation; set `action: "continue_debate"`.

#### Round 2:
* **Advocate**: Rebutted the Skeptic's pericarditis theory, emphasizing that focal localized ST elevations with reciprocal changes favor acute vascular occlusion over diffuse pericarditis.
* **Skeptic**: Re-iterated that without Troponin lab confirmation or echocardiography, declaring an uncomplicated STEMI carries risk of missed non-coronary mimics.
* **Evidence-Checker**: Verified that clinical guidelines mandate emergency coronary evaluation but require biomarker sampling before definitive discharge or non-invasive triage.
* **Senior Moderator Final Verdict**:

```json
{
  "final_verdict": "Refer to Specialist / Inconclusive",
  "confidence_score": 0.62,
  "action": "finalize",
  "audit_trail": "Round 2 analysis: The ECG AI output assigns a 100% probability to acute myocardial infarction based on significant ST-segment elevation. This is high-strength electrophysiological evidence. However, WHO HEARTS guidelines require a rise/fall in cardiac biomarkers (Troponin) alongside ischemic ECG changes. The chest radiograph excludes gross mimics (no pneumothorax, effusion, or cardiomegaly), and the dermoscopy study confirms an incidental benign melanocytic nevus. Because lab troponins are absent and conflicting non-coronary mimics cannot be definitively excluded, the case is referred immediately to cardiology/specialist care."
}
```

### Key Validations
* ✅ **Multi-Turn Cycling**: The debate successfully cycled across multiple rounds with accumulated `debate_history`.
* ✅ **Cross-Modal Clinical Reasoning**: The LLM understood that the skin lesion was incidental to the acute cardiac finding and that the clear X-ray helped rule out thoracic mimics.
* ✅ **Zero Rate Limits or Token Errors**: Finished with 100% complete responses using the newly optimized token windows (`max_tokens=1024`, fast RAG query extractor).
