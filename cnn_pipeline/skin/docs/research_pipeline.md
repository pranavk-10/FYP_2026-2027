# HAM10000 ResNet18 research pipeline

This extends the existing skin classifier and `DiagnosticState` interface. ResNet18, the seven-class mapping, lesion grouping, weighted cross-entropy and clean preprocessing are retained. `weights/model_ham10000_best.pth` is protected; upgraded models use separate filenames.

Locally, metadata lists 10,015 images but only 189 images (185 lesions) are present. The current model comes from a 30-epoch run on that available subset. Earlier smoke tests are archived separately. The existing baseline has no recorded training split; evaluating it on a new partition cannot establish unbiased held-out performance.

## Architecture

| File | Responsibility |
|---|---|
| `train/train_ham10000.py` | Existing ResNet18 training, Adam, scheduler, weighted CE and validation checkpoint selection |
| `train/download_ham10000.py` | Existing downloader; default destination corrected to `skin/data/` |
| `core/common.py` | Original class/preprocessing contracts, seeds, JSON, checkpoint loading and baseline protection |
| `core/data_pipeline.py` | Dataset validation and lesion-disjoint split manifests |
| `core/augmentation.py` | Training-only augmentation and unchanged clean transforms |
| `core/calibration.py` | Validation-only temperature fitting and calibration metrics/plots |
| `research/clinical_metrics.py` | Clinical metrics, operating points, ROC/PR and decision curves |
| `research/statistical_tests.py` | Paired McNemar and lesion-cluster bootstrap |
| `research/evaluation.py` | Calibration, clinical evaluation and model-comparison implementation |
| `research/robustness.py` | Ten synthetic disturbances and optional FGSM/PGD implementation |
| `evaluation.py`, `robustness.py` | Thin compatibility entry points for existing CLI commands |
| `skin_inference.py`, `__init__.py` | Existing public inference interface and JSON schema |
| `tests/` | Contract/mathematical tests and a real-image integration smoke runner |

```text
data/HAM10000_metadata.csv + data/images/*.jpg
  -> validated metadata -> persisted lesion-disjoint splits.json
  -> training: augmentation + weighted CE + validation checkpoint selection
  -> validation: temperature fitting + operating-threshold selection
  -> test: clinical metrics + calibration analysis + paired statistics + robustness
  -> inference: original and temperature-scaled softmax in DiagnosticState
```

Every lesion belongs to exactly one split. Splitting is stratified at the lesion level; defaults are 70%/15%/15% of lesions. Image fractions can differ. If class counts/split sizes prevent stratification, a reproducible unstratified lesion split is used with explicit logged limitations. Missing classes are reported, and training refuses a training partition missing any diagnostic category. Conflicting labels within a lesion, duplicate image IDs, unknown classes and null grouping IDs are rejected.

`splits.json` records image/lesion/label assignments, a dataset fingerprint, seed, fractions, class counts and limitations. Evaluation rejects changed available metadata, missing images, altered partitions and structured checkpoints trained with a different manifest. Keep original image files and environment for reproduction: the fingerprint identifies metadata records, not pixel bytes. All evaluation sample ordering follows the manifest. Test data is never used for fitting, checkpoint/policy/hyperparameter selection, calibration or threshold selection.

## Installation

Use Python 3.11+ in an isolated environment with matching PyTorch/torchvision versions. Install a PyTorch build appropriate for CPU/CUDA, then:

```powershell
python -m pip install -r requirements.txt
```

Research dependencies are pandas, scikit-learn, SciPy and matplotlib, alongside existing PyTorch, torchvision, Pillow and NumPy. The original trainer already imported pandas/scikit-learn. Downloader dependencies are tqdm and huggingface-hub. No Albumentations, OpenCV, statsmodels or attack toolkit is needed. SciPy provides mild morphology/deformation, scalar optimization and exact statistical tests.

Run these commands from `cnn_pipeline/skin/`. Package invocation from the repository root also works, for example `python -m cnn_pipeline.skin.train.train_ham10000` and `python -m cnn_pipeline.skin.evaluation clinical ...`. CPU is supported; CUDA is selected automatically when available or with `--device cuda`.

## Training and augmentation

```powershell
python train/download_ham10000.py --dest data
python train/train_ham10000.py --data-dir data --output weights/model_ham10000_upgraded.pth --results-dir results/experiment01 --epochs 30 --batch-size 32 --lr 0.0001 --seed 42 --validation-size 0.15 --test-size 0.15 --augmentation custom
```

ImageNet initialization remains the default. Use `--no-pretrained` for offline smoke tests. Selection uses validation macro-F1, with validation accuracy as a tie-breaker. Inverse-frequency class weights use training counts only. The first epoch saves a checkpoint even with zero F1. Existing checkpoints or experiment directories are rejected to avoid silent overwrites.

Input remains 224x224 RGB with the original ImageNet mean `[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. Clean resizing retains torchvision's original bilinear behavior. Select exactly one policy:

| `--augmentation` | Behavior |
|---|---|
| `baseline` | Original flips, ±20° rotation and ColorJitter (.1 brightness/contrast/saturation) |
| `none` | Deterministic clean preprocessing |
| `custom` (default) | Flips, ±15° bilinear rotation, safe photometric operations, mild morphology/elastic deformation |
| `randaugment` | Controlled adaptation: two safe photometric operations at fixed restrained magnitude |
| `autoaugment` | Controlled adaptation: one of four predefined safe pairs, applied probabilistically |
| `trivialaugment` | Controlled adaptation: one safe operation with random bounded magnitude |

These are **dermoscopy adaptations**, not unrestricted published/ImageNet torchvision policies. Solarization, posterization, cutout and aggressive warping are excluded. Photometric limits are ±12% brightness/contrast/saturation, hue ±.015, gamma .9–1.1, illumination gain/gradient up to .08 and red/blue color-temperature gain shifts up to .035. Custom operations each have probability .35.

Morphology randomly selects dilation, erosion, opening or closing with a 3x3 luminance kernel, blending only 15% of its luminance change while preserving chroma. Elastic fields use Gaussian smoothing (sigma 12px), a maximum 1.5px displacement at 224x224, bilinear interpolation and reflected boundaries. Default probabilities are .12 and .15 respectively. Disable with `--morphology-probability 0 --elastic-probability 0`. These operations apply only in advanced training policies; validation, clean test, `baseline` and `none` never receive them.

The bounds are engineering safeguards, not clinically validated safety guarantees. Review augmented examples with domain experts and select policies on training/validation before final test evaluation.

Configuration records seed, UTC timestamp, model version, preprocessing, augmentation, optimizer settings, device, Python/package versions and split identity. Deterministic algorithms, dataloader generator/worker seeds and CUDA `CUBLAS_WORKSPACE_CONFIG=:4096:8` are enabled; unsupported deterministic operations raise errors. Identical results across different hardware/library versions are not guaranteed.

Structured checkpoints contain model/optimizer/scheduler state, selected epoch, best validation metrics, mappings, preprocessing, configuration, seed, model version, split hash and an initially empty calibration-temperature field. Calibration is stored separately after fitting the selected checkpoint.

## Calibration

```powershell
python evaluation.py calibrate --checkpoint weights/model_ham10000_upgraded.pth --splits results/experiment01/splits.json --data-dir data --output weights/calibration_temperature.json --results-dir results/experiment01/calibration --n-bins 15
```

Only **validation logits** fit the positive scalar T by minimizing unweighted multiclass NLL. Log-temperature bounds are [-3,3] (T about .05–20.09); bound hits are reported. T remains 1 if optimization would worsen NLL. Weights are unchanged. ECE is not guaranteed to improve when optimizing NLL.

The artifact records T, validation count, optimizer details, mappings, checkpoint/split SHA256, timestamp and before/after metrics. Mismatched artifacts are rejected. The command also saves validation-selected operating thresholds, logits/predictions, top-label reliability and seven class-wise diagrams.

ECE is `sum_b (n_b/N) * abs(accuracy_b - confidence_b)`, using confidence `max(p)` and correctness `argmax(p)==y`. Equal-width bins are `[0,1/B]`, `(1/B,2/B]`, ..., `((B-1)/B,1]`. Empty bins contribute zero. MCE is the largest occupied-bin gap. Class-wise diagrams/ECE use `p_c` versus the indicator `y==c`. Bin counts, boundaries and means are recorded. Choose bin count before test evaluation; sparse bins can be unstable.

| Metric | Meaning and medical relevance |
|---|---|
| ECE / MCE | Mean / worst bin confidence-accuracy gap; reveals misleading confidence |
| Brier score | Mean sum of seven squared probability errors, range 0–2; evaluates the full distribution |
| NLL | Mean negative log true-class probability; penalizes confident errors |
| Reliability diagram | Confidence versus observed accuracy/frequency |

T>0 preserves an image's class ranking and argmax. Calibration is not verification of an individual patient's clinical risk, especially under distribution shift.

## Clinical evaluation

```powershell
python evaluation.py clinical --checkpoint weights/model_ham10000_upgraded.pth --splits results/experiment01/splits.json --data-dir data --calibration weights/calibration_temperature.json --thresholds results/experiment01/calibration/metrics/operating_points.json --results-dir results/experiment01/clinical --n-bins 15 --bootstrap 1000 --seed 42
```

If `--thresholds` is omitted, operating thresholds are selected on validation before test evaluation. Targets default to .90/.95/.99, configurable with `--specificity-targets` and `--sensitivity-targets`. Candidates are unique validation scores, 0 and a sentinel just above 1 (predict none). Among candidates satisfying a validation constraint, maximize the complementary metric, then the constraint and threshold to break ties. Report achieved test sensitivity/specificity: validation's target may not hold on test, particularly with few negatives.

| Metric / analysis | Meaning and medical relevance |
|---|---|
| Sensitivity | TP/(TP+FN); proportion of a diagnostic category detected |
| Specificity | TN/(TN+FP); proportion of other categories correctly excluded |
| Precision | TP/(TP+FP); fraction of category predictions that are correct |
| Balanced accuracy | Mean recall of evaluation classes present; reduces majority-class dominance |
| Macro-F1 / weighted-F1 | Harmonic precision/recall averaged equally / by true support |
| Sensitivity at specificity targets | Detection under a prespecified false-positive constraint |
| Specificity at sensitivity targets | False-positive control under a prespecified detection constraint |
| ROC / AUC | Per-class one-vs-rest discrimination, macro and micro averages |
| PR / average precision | Precision-recall tradeoff, useful under class imbalance |
| Decision curves | Exploratory one-vs-rest net benefit against treat-all/treat-none |

For category c, c is positive and six other categories are negative. Independent OvR operating decisions are not a seven-way argmax decision policy. Macro sensitivity/specificity average defined values. Macro-F1/precision use all seven classes, with zero undefined F1/precision. Missing positives/negatives yield null sensitivity/specificity/AUC/AP as appropriate. Macro AUC/AP average defined classes; valid AUC class count is reported. Micro AUC flattens seven one-hot targets and probabilities. Undefined statistics serialize to JSON null.

Decision curves use `NB = TP/N - FP/N * t/(1-t)` for threshold probabilities .01–.80. Treat-all is prevalence minus `(1-prevalence)*t/(1-t)`; treat-none is zero. Each category is a hypothetical separate binary clinical decision. The grid is descriptive, and no clinical threshold is selected from test curves. Numeric JSON/CSV and plots are saved. These analyses do not define a validated intervention or prove improved patient outcomes.

## Paired comparison and statistical testing

```powershell
python evaluation.py compare --baseline weights/model_ham10000_best.pth --checkpoint weights/model_ham10000_upgraded.pth --splits results/experiment01/splits.json --data-dir data --calibration weights/calibration_temperature.json --results-dir results/experiment01/comparison --bootstrap 1000 --seed 42 --allow-unverified-baseline
```

The final flag marks the raw baseline's unknown split provenance. Results are descriptive; no unbiased improvement claim is supported. Other legacy evaluations require `--allow-unverified-checkpoint`. These flags cannot bypass structured-checkpoint split mismatches. The upgraded model is not initialized from potentially overlapping legacy baseline training.

For a valid comparison, retrain the baseline using its original augmentation, the same predeclared split and training budget, preserving original weights:

```powershell
python train/train_ham10000.py --data-dir data --output weights/model_ham10000_baseline_retrained.pth --results-dir results/baseline01 --splits results/experiment01/splits.json --epochs 30 --batch-size 32 --lr 0.0001 --augmentation baseline
python evaluation.py calibrate --checkpoint weights/model_ham10000_baseline_retrained.pth --splits results/baseline01/splits.json --data-dir data --results-dir results/baseline01/calibration
python evaluation.py compare --baseline weights/model_ham10000_baseline_retrained.pth --baseline-calibration weights/model_ham10000_baseline_retrained.calibration.json --checkpoint weights/model_ham10000_upgraded.pth --calibration weights/calibration_temperature.json --splits results/experiment01/splits.json --data-dir data --results-dir results/experiment01/paired_comparison --bootstrap 1000 --seed 42
```

Both models predict exactly the same ordered test images. McNemar reports `[[both correct, baseline only correct], [upgraded only correct, both wrong]]`, a continuity-corrected chi-square statistic, **exact two-sided binomial p-value**, asymptotic p-value and significance at alpha .05. With no discordant pairs, statistic=0 and p=1. Image-paired McNemar assumes independent pairs; multiple images per lesion can violate this.

The preferred uncertainty analysis resamples lesion IDs with replacement, retaining all images of every sampled lesion (including repeated clusters). Metrics remain image-weighted. Both models use the same bootstrap draws, preserving pairs. Percentile 95% intervals cover accuracy, macro-F1, macro-AUC, macro and per-class sensitivity/specificity, ECE and Brier, plus paired differences. Undefined class values are excluded; valid/invalid counts are reported. Macro AUC/sensitivity/specificity replicates missing original evaluation classes are excluded to avoid changing the population; macro-F1 always uses seven classes. Real analysis should use at least 1000 draws. Bootstrap is opt-in; smoke uses 30 draws only.

The comparison table includes accuracy, macro-F1, macro-AUC, sensitivity, specificity, ECE and Brier. Table improvement is direction-adjusted (positive means better, including lower ECE/Brier); CI deltas retain the literal upgraded-minus-baseline sign. A supported accuracy-improvement flag requires both checkpoint split provenances verified, higher upgraded accuracy, significant McNemar and a lesion-bootstrap accuracy-delta lower bound above zero. Other metrics need their own analysis. Multiple metric comparisons are exploratory and not multiplicity-adjusted.

## Robustness

```powershell
python robustness.py --checkpoint weights/model_ham10000_upgraded.pth --splits results/experiment01/splits.json --data-dir data --calibration weights/calibration_temperature.json --results-dir results/experiment01/robustness --severity 1 --seed 42
python robustness.py --checkpoint weights/model_ham10000_upgraded.pth --splits results/experiment01/splits.json --data-dir data --calibration weights/calibration_temperature.json --results-dir results/experiment01/robustness_attacks --corruptions hair illumination noise jpeg --attacks fgsm pgd --epsilon 0.0078431373 --iterations 10 --seed 42
```

Disturbances are `hair`, `illumination`, `brightness`, `contrast`, `noise`, `blur`, `jpeg`, `color`, `vignetting`, `uneven_lighting`. Select a subset with `--corruptions`; severity is 1–5. Seeds derive from seed/image ID/corruption, independent of sample order and batch size. Original images are disturbed before deterministic resize/normalization; clean images remain unchanged and available.

Hairs are curved quadratic strands with coherent but varied direction, brown tones, variable width, supersampled antialiasing and soft opacity. They approximate hair occlusion; realism has not been clinically validated. Other disturbances use intensity scaling, Gaussian noise/blur, JPEG quality changes, red/blue gains, peripheral vignetting and spatial lighting gradients. Examples are saved for inspection.

Every condition reports accuracy, macro-F1, macro-AUC, seven sensitivities, ECE and degradation from clean. Degradation is clean-minus-disturbed for performance, disturbed-minus-clean for ECE; positive means worse. Tables/plots distinguish synthetic and adversarial conditions. The clean validation temperature is reused without fitting on corrupted test data.

FGSM/PGD are optional white-box untargeted cross-entropy attacks. Inputs are converted from ImageNet-normalized tensors to [0,1] RGB pixels, and gradients pass through normalization. Perturbations are projected into the L-infinity epsilon ball and valid pixel bounds. Only input gradients are requested; no optimizer/weight updates occur. Epsilon uses [0,1] units (2/255 approximately .007843); the report also gives 0–255 units. FGSM uses one epsilon step. PGD uses random initialization, configurable iterations and `--step-size` (default `2*epsilon/iterations`). These attacks do not certify robustness. Robustness reports point estimates; bootstrap options belong to clinical/comparison evaluation.

## Inference compatibility

`predict_skin(image_input, model_path=None, device=None)` and `load_skin_model(model_path=None, device=None)` retain signatures and exports. Inputs remain paths, PIL images or bytes. Default loading prefers `weights/model_ham10000_upgraded.pth`, then original baseline locations. Raw state dictionaries and structured `model_state_dict`, `state_dict` or `model` dictionaries are supported with strict shape/mapping checks. Whole pickled model objects are not loaded; the actual repository baseline is a compatible raw state dictionary.

```powershell
python skin_inference.py data/images/ISIC_0024306.jpg
```

From the repository root:

```python
from cnn_pipeline.skin import predict_skin, load_skin_model
result = predict_skin('patient_lesion.jpg', device='cpu')
```

The conventional upgraded filename reads adjacent `calibration_temperature.json`. Other checkpoint filenames read `<stem>.calibration.json`. Set `SKIN_CALIBRATION_PATH` for an explicit artifact without changing signatures. A missing default artifact means T=1. Explicitly missing, invalid or mismatched artifacts raise errors. Cache identity includes resolved path, device and model/artifact modification time.

All original DiagnosticState keys remain: `modality`, `input_type`, `prediction`, `differential`, `evidence`, `explainability`. Prediction retains `primary_diagnosis`, `raw_class`, `probability`, `calibrated_probability`, `findings`. `findings` and `differential` contain the same seven-item list, retaining each item's `label`, `raw_class`, `probability`, `calibrated_probability`.

`probability` is original softmax, `calibrated_probability` is softmax(logits/T). Both distributions sum to approximately one; sorting uses scaled probabilities. T>0 preserves ranking. Without an artifact, T=1 and both fields are **uncalibrated softmax**, not empirically calibrated confidence. The unchanged order is `0 akiec`, `1 bcc`, `2 bkl`, `3 df`, `4 mel`, `5 nv`, `6 vasc`.

The classifier does not establish specific dermoscopic observations or exclude pathology, so evidence lists are empty. Existing Grad-CAM method and `resnet18.layer4` target metadata remain. No Grad-CAM computation was found in the inspected repository; region text now states that no heatmap/localization was computed. External hooks can still target the same layer. See [inference_schema.md](inference_schema.md) for the current output contract; the obsolete training-folder schema document has been removed.

## Outputs and testing

Runs use dedicated `results/<experiment>/` directories (UTC timestamp by default):

| Location | Contents |
|---|---|
| `config.json`, `splits.json`, `training.log` | Configuration, exact partitions/class counts and training log |
| `calibration/metrics`, `calibration/plots` | Validation predictions, calibration metrics, operating thresholds and reliability diagrams |
| `clinical/metrics`, `clinical/plots` | Test metrics/predictions, bootstrap intervals, ROC/PR/reliability and decision curves |
| `comparison/reports`, `comparison/metrics` | JSON/CSV/Markdown comparison, McNemar/paired intervals and paired predictions |
| `robustness/metrics`, `robustness/reports`, `robustness/plots`, `robustness/examples` | Disturbed metrics/predictions, comparison CSV, plots and image examples |

Prediction NPZ files include logits, labels, raw/scaled distributions, image/lesion IDs, class ordering and T. JSON uses null for undefined values. Evaluation config records arguments and checkpoint/split identities. Expensive bootstrap/attacks are optional. Existing checkpoint/result/calibration artifacts are not silently overwritten; use fresh experiment names. Generated results and upgraded weights are gitignored.

```powershell
python -m unittest discover -s tests -p test_research_pipeline.py -v
python tests/smoke_experiment.py --results-dir results/smoke_example
```

Tests cover dataset/classes, leakage/manifest tampering, sparse-split limitations, all augmentation policies, deterministic clean transforms, hand-computable calibration/clinical metrics, validation-only fitting/threshold guards, exact McNemar, paired cluster bootstrap, every corruption, FGSM/PGD pixel constraints/weight integrity, legacy checkpoint loading and DiagnosticState keys. The smoke runner copies eight local lesions per class, trains one epoch without pretraining on CPU, saves/reloads a structured checkpoint, calibrates/evaluates, compares legacy weights descriptively, runs all ten corruptions and both attacks, and verifies unchanged baseline SHA256.

The earlier measured smoke run is documented in [smoke_results.md](smoke_results.md). Complete outputs are preserved in `results/archive/smoke_final.zip` and `results/archive/smoke_verified.zip`, with verified file hashes in `results/archive/index.json`. Restore an archive into `results/` to recover the original paths. Robustness evaluation hashes model parameters and buffers before/after; repeated smoke training produced identical model-state hashes and split assignments.

The later 30-epoch ImageNet-initialized run on all 189 available images is in `results/training_20261031/`. The selected epoch-8 checkpoint remains in `weights/model_ham10000_upgraded.pth`, with adjacent calibration JSON. On 28 held-out images it achieved accuracy .75, macro-F1 .7056 and macro-AUC .9386. These are local-subset measurements. Baseline provenance is unknown, so the numeric comparison does not establish statistically supported improvement. Training/validation curves show overfitting; checkpoint selection retained the earlier best validation model.

Remaining research limitations: full dataset/baseline split provenance are unavailable locally; sparse validation/test sets cannot support precise high-specificity estimates; augmentation/artifact realism needs clinical review; heatmaps/verified findings are not computed; no external validation, repeated-seed research study, clinical outcome study, subgroup/fairness analysis or robustness certification was performed. A full-dataset and provenance-verified paired experiment is still required for research claims.
