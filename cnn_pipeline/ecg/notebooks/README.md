# ECG notebooks

| Notebook | Model | What it contains |
|---|---|---|
| `v1_resnet18_mendeley.ipynb` | v1, ResNet18, 4 classes | Original model trained on the Mendeley ECG images (random split). Kept for reference; superseded by v3 |
| `v3_01_data_prep_and_stage1_training.ipynb` | v3, ConvNeXt-Tiny | Data preparation, image cache, data loading, GPU check, stage 1 training, stage 2 epochs 1–6 |
| `v3_02_stage2_evaluation_and_deploy.ipynb` | v3, ConvNeXt-Tiny + ResNet18 baseline | Stage 2 epochs 7–12, test evaluation, ResNet18 baseline, significance tests, Grad-CAM, external Mendeley test, deploy package |

The v3 work ran on free Google Colab (T4 GPU). It is split over two notebooks because the first Google account reached its daily GPU limit; the second notebook was run from another account sharing the same Drive folder, and resumed training from the saved checkpoint.

Both notebooks contain their printed outputs, so results can be read without rerunning anything.

---

## Data and storage

Everything is read from and written to Google Drive under `MyDrive/ecg_v3/`:

| Path | Contents |
|---|---|
| `genecg_A.tar`, `genecg_B_subset.tar` | GenECG images (Hugging Face `edcci/GenECG`): Dataset A clean, Dataset B photo-like subset |
| `cache_704x512.tar` | All 29,177 images letterboxed to 704×512 (used by every training/evaluation cell) |
| `labels/` | `labels_genecg.csv` (PTB-XL labels, folds, patients), `subset_B.csv`, PTB-XL tables |
| `checkpoints/` | Best and last checkpoints plus per-epoch history for all four runs |
| `results/` | Test tables, paired comparisons, external results, calibration, saved logits, Grad-CAM figure |
| `external/` | Mendeley zip + list of 491 unique images (external test) |
| `deploy/` | Inference module, deploy weights, sample outputs, module checks |

Labels come from PTB-XL v1.0.3 (5 diagnostic superclasses, multi-label) and the official patient-level folds: 1–8 train, 9 validation, 10 test.

---

## Notebook 1: `v3_01_data_prep_and_stage1_training.ipynb`

| Code cells | Step |
|---|---|
| 1 | Mount Drive, set paths and options |
| 2 | Download the PTB-XL label tables |
| 3 | Map SCP codes to 5 superclasses; save `labels_genecg.csv` |
| 4–5 | Download GenECG Dataset A (21,799 images) and archive it to Drive |
| 6–8 | Hugging Face login (token from Colab secrets); download a Dataset B subset (all of folds 9–10 + 3,000 from folds 1–8) as 1600 px JPEG and archive it |
| 9–10 | Sample image grid and image size check |
| 11 | Session start: unpack the archives |
| 12 | Build the 704×512 letterboxed cache and save it to Drive |
| 13 | Session start for training: unpack the cache, check counts |
| 14 | Splits, patient-overlap check, dataset class, GPU augmentation, loader speed test |
| 15 | Fix: augmentation always in full precision |
| 16 | Load ConvNeXt-Tiny and ResNet18, memory and speed test |
| 17 | Training function (per-epoch checkpoints, resume) and **stage 1**: ConvNeXt-Tiny on clean images, 10 epochs |
| 18 | **Stage 2**: 50/50 photo + clean images (stopped after epoch 6 at the GPU limit) |

## Notebook 2: `v3_02_stage2_evaluation_and_deploy.ipynb`

| Code cell | Step |
|---|---|
| 1 (R1) | Full session setup: mount Drive, unpack cache, define data, augmentation, models, training and evaluation functions. **Run first after any reconnect** |
| 2 (R2) | Resume stage 2 from the saved checkpoint, epochs 7–12 |
| 3 (R3) | Test-set evaluation, temperature scaling, thresholds |
| 4 (R4) | ResNet18 baseline, both stages, plus the 8-row test table |
| 5 (R5) | Patient-level paired bootstrap tests and the two-model ensemble |
| 6 (R6) | Grad-CAM check and export of `ecg_convnext_v3_deploy.pth` |
| 7 (R7) | External test on 491 real Mendeley printouts |
| 8–9 (D1–D2) | Create the deploy folder and write `ecg_inference.py` |
| 10–11 (D3–D4) | Test the module on sample images and on every test image |
| 12 (D5) | Referral-rule update in the module |
| 13 (D6) | Module behaviour on the Mendeley printouts |

Note: the ECG evidence wording in `cnn_pipeline/ecg/ecg_inference.py` was later changed in the repo (class-only descriptions, no wave-level features). The model, preprocessing and scores are identical to the notebook version.

---

## Headline results (PTB-XL fold 10 test unless noted)

| | Macro-AUROC (95% CI) |
|---|---|
| Final model, clean images | 0.931 (0.924–0.938) |
| Final model, photo-like images | 0.917 (0.909–0.925) |
| External: real Mendeley printouts, MI vs normal | 0.947 (0.919–0.972) |

Full write-up: `ecg_updated/docs/ECG_Model_Report.docx` and `ecg_updated/docs/ECG_V3_PLAN.md`.

## Rerunning

1. Open the notebook in Colab with a T4 GPU runtime and the `ecg_v3` folder in your Drive (or a shortcut to it in My Drive).
2. Notebook 1 only needs to be rerun to rebuild the data; its outputs are already on Drive.
3. In notebook 2, run cell R1 first, then any later cell. Training cells resume from the last saved epoch, and evaluation cells reuse saved logits.
