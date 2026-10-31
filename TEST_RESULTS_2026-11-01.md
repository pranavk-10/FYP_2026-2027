# System Test Results: 2026-11-01

**Compared with**: `TEST_RESULTS.md` (2026-10-08)
**How run**: every request went through the real FastAPI endpoints (in-process `TestClient`, same code as `uvicorn`) with the real Groq LLMs, Pinecone RAG and Tesseract OCR. Test files: `testing/` (`ecg.jpg`, `xray.jpeg`, `skin.jpeg`, `cbc-report-format.pdf`). Debates used `max_rounds = 2`.

**System version under test**

| Part | Version |
|---|---|
| ECG model | ConvNeXt-Tiny v3, 5 PTB-XL superclasses, calibrated (T = 1.38) |
| X-ray model | DenseNet121 (unchanged by this work) |
| Skin model | ResNet18 `model_ham10000_upgraded.pth` (teammate's latest commit) |
| Debate | LangGraph Advocate → Skeptic → Evidence-Checker → Moderator, `openai/gpt-oss-120b` on Groq |

Two debate runs are reported:
- **Run A**: after the first debate fixes (round-limit stop, grounding rule, top-3 differential in the summary, moderator notes passed to the next round, per-modality guideline retrieval).
- **Run B (current code)**: after the second fixes (no wave-level wording in ECG evidence, stricter grounding, per-modality assessment with `secondary_findings`, new moderator rules, confidence capped at the model probability, "acute" MI guard, verdict-card update).

---

## Executive summary

| # | Test | Endpoint | Status | Run B result |
|---|---|---|:---:|---|
| 1 | Blood report OCR | `/api/ocr/blood-report` | 200 | 13 CBC values + demographics; Hb, WBC, platelets flagged normal |
| 2 | Skin + debate | `/api/analyze/skin` | 200 | Melanocytic Nevus, 0.97 |
| 3 | ECG prediction | `/api/predict/ecg` | 200 | MI 97.8%, Conduction Disturbance 63.7% (detected), Ventricular Hypertrophy 15.6% (watch-list) |
| 3b | ECG + debate *(new)* | `/api/analyze/ecg` | 200 | Myocardial Infarction, 0.97 |
| 4 | X-ray prediction | `/api/predict/xray` | 200 | Cardiomegaly 72.0%, normality 26.1% |
| 4b | X-ray + debate *(new)* | `/api/analyze/xray` | 200 | Cardiomegaly, 0.72 |
| 5 | ECG + X-ray + skin | `/api/analyze/multimodal` | 200 | Myocardial Infarction, 0.97 + cardiomegaly, nevus as secondary findings |
| 5b | ECG + X-ray *(new)* | `/api/analyze/multimodal` | 200 | Myocardial Infarction, 0.97 + cardiomegaly |
| 5c | ECG + skin *(new)* | `/api/analyze/multimodal` | 200 | Myocardial Infarction, 0.95 + nevus |
| 5d | X-ray + skin *(new)* | `/api/analyze/multimodal` | 200 | Cardiomegaly, 0.72 + nevus |
| 5e | ECG + X-ray + skin + blood report *(new)* | `/api/analyze/multimodal` | 200 | Myocardial Infarction, 0.97 + cardiomegaly, conduction disturbance, nevus |

**11 / 11 requests returned HTTP 200.** No crashes, no rate-limit errors, every debate ended within 2 rounds.

---

## Debate verdicts: October vs Run A vs Run B

| Test | 2026-10-08 | Run A | Run B (current) |
|---|---|---|---|
| 2. Skin | Benign nevus, 0.93 | Nevus, 0.97 | Nevus, 0.97 |
| 3b. ECG alone | – | Refer / Inconclusive, 0.45 (2 rounds) | **MI, 0.97** (1 round) |
| 4b. X-ray alone | – | Cardiomegaly, 0.78 | Cardiomegaly, **0.72** (capped) |
| 5. ECG + X-ray + skin | Refer / Inconclusive, 0.62 | Refer / Inconclusive, 0.62 (2 rounds) | **MI, 0.97** (1 round) |
| 5b. ECG + X-ray | – | "Acute" MI, 0.86 | **MI, 0.97** |
| 5c. ECG + skin | – | "Acute" MI, 0.95 | **MI, 0.95** |
| 5d. X-ray + skin | – | Cardiomegaly, 0.78 | Cardiomegaly, 0.72 |
| 5e. All + blood report | – | Refer / Inconclusive, 0.78 (2 rounds) | **MI, 0.97** (1 round) |

### Quality checks on the debate transcripts

| Check | October | Run A | Run B |
|---|---|---|---|
| Same ECG gives the same verdict in every combination | – | No (Refer 0.45 / MI 0.86 / MI 0.95 / Refer 0.62) | **Yes** (MI 0.95–0.97) |
| Verdict says "acute" MI although the model cannot tell acute from old | ECG text said "ST elevation" | 2 of 5 ECG debates | **0** |
| Invented skin features (pigment network, globules, borders) stated as observed | 4 claims | 0 | 0 |
| Agents claiming the ECG model "shows" Q-waves / ST changes | "focal ST elevations with reciprocal changes" | 1–5 per ECG debate | **0** |
| X-ray finding reported when an ECG is also uploaded | – | Never (only in the audit trail as "non-specific") | **Always**, as a secondary finding with follow-up |
| Skin finding reported when other tests are uploaded | Incidental, in text | Incidental, in text | **Always**, as a secondary finding |
| Confidence above the model's own probability | – | Yes (0.78 for a 72% X-ray finding) | **No** (capped in code) |
| Guidelines retrieved per modality in multimodal cases | Mixed search | Yes, labelled per modality | Yes |

### Secondary findings (Run B, test 5e: all tests + blood report)

- XRAY: Cardiomegaly (0.72), obtain echocardiography and clinical correlation
- ECG: Conduction Disturbance (0.637), cardiology review, consider repeat ECG
- DERMOSCOPY: Melanocytic Nevus (0.975), routine skin examination, no urgent intervention

The audit trail treats the MI result as a screening alert. It states that the model cannot distinguish acute from old MI, and lists confirmatory steps: serial troponin, repeat ECG, cardiology review.

### Debate time (seconds)

| Test | Run A | Run B |
|---|---:|---:|
| 2. Skin | 47.2 | 12.7 |
| 3b. ECG | 104.0 | 58.4 |
| 4b. X-ray | 25.1 | 58.6 |
| 5. ECG + X-ray + skin | 134.7 | 67.9 |
| 5b. ECG + X-ray | 42.9 | 70.1 |
| 5c. ECG + skin | 54.1 | 61.5 |
| 5d. X-ray + skin | 47.8 | 67.2 |
| 5e. All + blood report | 162.7 | 66.3 |

Two-round debates in Run A took the longest. In Run B every debate finished in round 1 (about 60–70 s). Single-round times vary with LLM response length.

---

## CNN outputs (unchanged between runs)

**ECG (`testing/ecg.jpg`)**

| Class | Calibrated probability | Tier |
|---|---:|---|
| Myocardial Infarction | 0.978 | detected |
| Conduction Disturbance | 0.637 | detected |
| Ventricular Hypertrophy | 0.156 | watch-list |
| ST/T-wave Changes | 0.068 | unlikely |
| Normal ECG | 0.002 | unlikely |

October's ResNet18 v1 reported MI 100% with a fixed text "ST-segment elevation"; v3 reports calibrated probabilities and says waves were not measured.

**X-ray (`testing/xray.jpeg`)**: Cardiomegaly 0.72 (detected); Atelectasis 0.03, Pulmonary Infiltration 0.008, Pleural Thickening 0.008, Lung Consolidation 0.005; normality 0.261.
October's run reported "No Finding" for the same image. The X-ray module was not changed in this work, so the difference comes from changes in the X-ray code or weights since then.

**Skin (`testing/skin.jpeg`)**: Melanocytic Nevus 0.975; Melanoma 0.007, Dermatofibroma 0.007 (upgraded model; October's model gave 0.935).

**Blood report**: patient demographics (name, age 27, M) and 13 CBC values extracted; Hb 15 g/dL, WBC 5100 /µL, platelets 350,000 /µL, all flagged normal (same as October).

---

## Code changes behind Run B (not yet committed)

| File | Change |
|---|---|
| `multi_agent/graph.py` | Hard stop when `current_round > max_rounds` (prevents endless loops) |
| `multi_agent/state.py` | Shared grounding rule; summary shows top-3 differential and model referral flag per modality; `ModeratorVerdict.secondary_findings` |
| `multi_agent/nodes/advocate.py` | Assess each modality on its own, report all relevant findings; grounding rule; reads moderator notes |
| `multi_agent/nodes/skeptic.py` | Uses top-3 differential and referral flags; challenges unsupported image features |
| `multi_agent/nodes/evidence.py` | Guideline retrieval per modality for multimodal cases; flags claims unsupported by model output |
| `multi_agent/nodes/moderator.py` | New decision rules (probability-based evidence, per-modality assessment, secondary findings, images-only context); final-round referral; notes passed to next round; confidence capped at model probability; "acute" MI guard |
| `cnn_pipeline/ecg/ecg_inference.py` | Evidence text names the class only, states that individual waves were not measured |
| `frontend/src/components/VerdictCard.jsx` | Shows "Other Findings"; next steps match the uploaded tests; model names updated |

Offline tests (fake LLM, no API calls) cover the round limit, summary format, moderator note hand-over, per-modality retrieval, confidence cap and "acute" guard; all pass. The frontend builds without errors.

---

## Known limitations and next steps

1. **X-ray evidence text still describes image features** (`cnn_pipeline/xray/xray_inference.py`, line 66: "Enlarged cardiac silhouette exceeding normal cardiothoracic ratio on PA view."). The Advocate repeated it as observed in test 4b; the Skeptic challenged it. Reword it to name the class only, as was done for ECG.
2. **The debate now follows strong CNN findings closely.** Findings with probability ≥ 0.70 and no referral flag are finalized in round 1; the Skeptic's points become next steps. Referral happens for weak or flagged findings or real conflicts. A debate-vs-CNN-alone evaluation on labelled cases is needed to show what the debate adds.
3. **Guideline coverage**: ECG retrieval uses WHO HEARTS (hypertension/CVD risk) and skin uses WHO skin NTDs; neither covers the CNN classes well.
4. **Skin model**: the upgraded checkpoint was trained on about 200 HAM10000 images (validation on 30); retraining on the full dataset is pending.
5. **Single run per test**: LLM output varies slightly between runs (e.g. 0.95 vs 0.97 for the same ECG).

Raw JSON responses for every test were saved during the run (not part of the repo).
