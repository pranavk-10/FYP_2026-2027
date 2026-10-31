# Pending Tasks

Last updated: 2026-11-01. Test status: `TEST_RESULTS_2026-11-01.md` (11/11 endpoint tests passing).
ECG model details: `ecg_updated/docs/ECG_V3_PLAN.md` and `cnn_pipeline/ecg/notebooks/README.md`.

---

## 🔴 High priority

### 1. Commit the current work (repository hygiene)
- The **frontend source has never been committed** (`frontend/src`, `package.json`, etc. are untracked). Commit it first.
- Track the new ECG weights with Git LFS before committing, like the skin model:
  `git lfs track "cnn_pipeline/ecg/weights/ecg_convnext_v3_deploy.pth"`
- Add `stuff/`, `scratch/` and `ecg_updated/deploy/weights/` (duplicate of the ECG weights) to `.gitignore`.
- Check that `frontend/public/samples/cbc-report-format.pdf` is a template, not a real patient's report, before pushing (it shows a patient name).
- Suggested commits: (1) ECG v3 model, (2) debate changes, (3) frontend, (4) docs and notebooks.

### 2. Evaluate the debate itself (needed for the paper)
- The CNNs are evaluated; the multi-agent debate is not. Nothing yet shows it adds value over the CNN alone.
- Benchmark on ~100 labelled cases per modality (ECG: PTB-XL fold 10 images + Mendeley printouts):
  **CNN alone** vs **single LLM call** vs **full debate**.
- Metrics: accuracy of the final verdict, referral rate, dangerous misses (e.g. MI called normal), time and API calls per case.
- Ablations: no Skeptic, no RAG, 1 vs 2 rounds.
- **Files**: new script, e.g. `evaluation/debate_benchmark.py`; uses `multi_agent/graph.py`.

### 3. Retrain the skin model on the full HAM10000 dataset
- The current `model_ham10000_upgraded.pth` was trained on about 200 of 10,015 images (checkpoint: 72 optimizer steps at batch 16) and calibrated on 30 validation images; calibration made ECE worse (0.224 → 0.272).
- Retrain on all images with lesion-level splits; refit the temperature on the full validation split; report test metrics with confidence intervals.
- Only a new `weights/model_ham10000_upgraded.pth` + matching `weights/calibration_temperature.json` are needed; no code change.
- **Files**: `cnn_pipeline/skin/train/train_ham10000.py`, `cnn_pipeline/skin/research/evaluation.py`

---

## 🟡 Medium priority

### 4. X-ray model: validation record and evidence wording
- Document how the DenseNet121 was trained and tested (dataset split, per-class AUROC). No validation numbers exist in the repo.
- Explain why `testing/xray.jpeg` changed from "No Finding" (2026-10-08) to "Cardiomegaly 72%" (2026-11-01).
- Reword `CLINICAL_FINDINGS_MAP` so it names the class only. Example: "Enlarged cardiac silhouette exceeding normal cardiothoracic ratio on PA view" is repeated by the agents as if it were measured on the image.
- **Files**: `cnn_pipeline/xray/xray_inference.py` (line ~66)

### 5. Skin output parity with ECG and X-ray
- Skin output has no `normality_score`, thresholds, risk tiers, `abstain` flag or evidence lines, so it can never trigger a referral and the agents get only one summary line for skin.
- Add tiers/thresholds (fitted on validation) and a malignant-class watch-list (e.g. melanoma, BCC above a low threshold).
- **Files**: `cnn_pipeline/skin/skin_inference.py`

### 6. RAG knowledge base matching the model classes
- Current documents: ECG → WHO HEARTS (hypertension/CVD risk), skin → WHO skin NTDs (leprosy, Buruli ulcer). Neither covers the CNN classes.
- Add: Fourth Universal Definition of MI and a standard ECG interpretation reference (conduction blocks, hypertrophy criteria); dermoscopy/melanoma guidance (ABCD rule, 7-point checklist). Check licences before ingesting.
- Make the Evidence-Checker mark each verdict as "from guidelines" or "general knowledge".
- **Files**: `multi_agent/rag_setup.py`, `multi_agent/nodes/evidence.py`

### 7. Debate improvements still open
- Use a different LLM for the Skeptic (all four roles currently use `openai/gpt-oss-120b`).
- Save each debate transcript (input, turns, verdict, model names, timestamp) as JSON for the audit trail and the paper.
- Optional patient context in the API (age, sex, main symptom).
- Structured outputs for Advocate / Skeptic / Evidence (formatting still drifts between runs).
- **Files**: `multi_agent/nodes/*.py`, `backend/api.py`

### 8. Grad-CAM for X-ray and skin
- ECG heatmaps work (`?gradcam=true` on `/api/predict/ecg`, `gradcam` form field on analyze endpoints).
- X-ray: TensorFlow Grad-CAM on DenseNet121 `conv5_block16`. Skin: PyTorch Grad-CAM on ResNet18 `layer4`.
- The frontend already shows an "Attention heatmap" toggle when `explainability.heatmap_png_base64` is present.

### 9. API polish
- Add a lightweight `/health` endpoint that reports whether each model loaded.
- Standardise error responses across single and multimodal endpoints.
- **Files**: `backend/api.py`

---

## 🟢 Low priority

### 10. ECG model extensions (paper extras)
- Robustness test: AUROC vs extra blur, rotation, JPEG compression and low resolution.
- Subgroup check: AUROC by sex and age group (saved logits on Drive, no GPU needed).
- Two more training seeds for stage 2 (mean ± SD).
- Rhythm classes (atrial fibrillation, tachycardia, bradycardia) from PTB-XL rhythm statements.
- Second real-paper test set: ECG-Image-Database (PhysioNet), fold-10 records only.

### 11. Production deployment
- Docker containerisation.
- Separate TensorFlow (X-ray) and PyTorch (ECG, skin) services if RAM is limited.
- Authentication, rate limiting and HTTPS.

---

## ✅ Completed

**ECG**
- [x] **ECG model v3**: ConvNeXt-Tiny, 5 PTB-XL superclasses, multi-label, trained on GenECG images with patient-level folds. Test macro-AUROC 0.931 (clean) / 0.917 (photo); MI vs normal on real Mendeley printouts 0.947. `cnn_pipeline/ecg/`
- [x] ECG calibration (temperature 1.38), per-class thresholds, risk tiers, normality score, referral flag (`abstain`)
- [x] ECG Grad-CAM heatmap (opt-in)
- [x] Mendeley leakage audit (928 files → 491 unique)
- [x] Old inference files moved out of the code folders (local copies in `stuff/old_inference/`). ECG v1 rollback from Git history: `git show 036e7e6:cnn_pipeline/ecg/ecg_inference.py > cnn_pipeline/ecg/ecg_inference.py` (v1 weights `model_ecg_image_best.pth` are still in the repo)
- [x] ECG notebooks organised in `cnn_pipeline/ecg/notebooks/` with a cell-by-cell README

**Other models and services**
- [x] X-ray CNN pipeline (DenseNet121, TensorFlow), Keras version mismatch fix — `cnn_pipeline/xray/`
- [x] Skin CNN pipeline (ResNet18, HAM10000) with research tooling — `cnn_pipeline/skin/`
- [x] Blood report OCR: Tesseract + Groq JSON schema + reference-range flags — `ocr_pipeline/`
- [x] FastAPI backend: predict and analyze endpoints for ECG, X-ray, skin; multimodal endpoint with blood report — `backend/api.py`
- [x] `requirements.txt` includes `timm` and `scipy`

**Debate engine** — `multi_agent/`
- [x] LangGraph debate (Advocate → Skeptic → Evidence-Checker → Moderator) with RAG over WHO guidelines in Pinecone
- [x] Hard round limit: the debate always ends at `max_rounds`; no consensus → "Refer to Specialist / Inconclusive"
- [x] Grounding rule in every agent: no image features stated as observed, no labels more specific than the model (e.g. no "acute" MI)
- [x] Summary shows top-3 differential and model referral flags for every modality
- [x] Moderator notes passed to the next round
- [x] Per-modality guideline retrieval for multimodal cases
- [x] Per-modality assessment with `secondary_findings` in the verdict (X-ray and skin findings no longer dropped when an ECG is present)
- [x] Confidence capped at the model's calibrated probability; consistent verdicts for the same ECG across combinations
- [x] Token and rate-limit optimisation, search-query extraction with `llama-3.1-8b-instant`

**Frontend** — `frontend/`
- [x] React + Vite clinical-report interface: case inputs with drag-and-drop, conclusion first, results by test, collapsible panel discussion (markdown), guideline passages with sources, print layout, honest progress timer
- [x] Backend address configurable with `VITE_API_URL` (`frontend/.env.example`)
