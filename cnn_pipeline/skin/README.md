# HAM10000 skin lesion pipeline

ResNet18 classification with seven unchanged classes, lesion-disjoint splitting, weighted cross-entropy, advanced augmentation, validation-only calibration and clinical/robustness evaluation.

## Layout

```text
skin/
  __init__.py                # Public predict_skin / load_skin_model exports
  skin_inference.py          # Existing DiagnosticState inference interface
  evaluation.py              # Compatibility CLI for research/evaluation.py
  robustness.py              # Compatibility CLI for research/robustness.py
  core/                      # Model contracts, transforms, datasets, calibration
  research/                  # Clinical metrics, statistics, evaluation, robustness
  train/                     # Training and dataset-download entry points
  tests/                     # Contract tests and optional integration smoke runner
  docs/                      # Research methodology, inference schema, historical results
  data/                      # HAM10000_metadata.csv and images/*.jpg
  weights/                   # Baseline, upgraded checkpoint, calibration JSON
  results/
    training_20261031/        # Current training/evaluation records and plots
    archive/                 # Verified ZIP archives of earlier smoke runs
  .research-deps/            # Ignored local packages required by this environment
  requirements.txt
```

Implementations live in `core/` and `research/`. Small root CLI files retain previous commands; the backend import `cnn_pipeline.skin.skin_inference.predict_skin` and public package exports remain unchanged. Data, checkpoint filenames and active experiment paths are unchanged.

## Setup and commands

Use an isolated Python 3.11+ environment with matching PyTorch/torchvision versions. Run from `cnn_pipeline/skin/`:

```powershell
python -m pip install -r requirements.txt
python train/download_ham10000.py --dest data
python train/train_ham10000.py --data-dir data --output weights/model_ham10000_upgraded_next.pth --results-dir results/experiment02 --epochs 30 --augmentation custom --seed 42
python evaluation.py calibrate --checkpoint weights/model_ham10000_upgraded_next.pth --splits results/experiment02/splits.json --data-dir data --results-dir results/experiment02/calibration
python evaluation.py clinical --checkpoint weights/model_ham10000_upgraded_next.pth --splits results/experiment02/splits.json --data-dir data --results-dir results/experiment02/clinical --bootstrap 1000
python robustness.py --checkpoint weights/model_ham10000_upgraded_next.pth --splits results/experiment02/splits.json --data-dir data --results-dir results/experiment02/robustness
python skin_inference.py data/images/ISIC_0024306.jpg
python -m unittest discover -s tests -p test_research_pipeline.py -v
```

The example uses a new filename to preserve current weights. Its temperature defaults to `weights/model_ham10000_upgraded_next.calibration.json`; evaluation and explicit-path inference find that sidecar automatically. CPU/CUDA are supported with `--device cpu` or `--device cuda`. Bootstrap and attacks are optional; add `--attacks fgsm pgd` to robustness evaluation for adversarial tests. Package commands such as `python -m cnn_pipeline.skin.research.evaluation clinical ...` work from the repository root; the previous `python -m cnn_pipeline.skin.evaluation ...` remains compatible.

This local environment uses the ignored `.research-deps/` directory for required analysis packages, so it was retained. To use those packages without installing into another environment:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) '.research-deps') + ';' + (Resolve-Path ../..).Path
& ../../.venv/Scripts/python.exe -m unittest discover -s tests -p test_research_pipeline.py -v
```

## Current trained model

- Weights: `weights/model_ham10000_upgraded.pth`.
- Calibration: `weights/calibration_temperature.json` (T = 1.027985).
- Training: `results/training_20261031/config.json`, `splits.json`, and `metrics/training_history.json`.
- Evaluation: `results/training_20261031/clinical/`, `comparison/`, and `robustness/`.
- Summary: `results/training_20261031/reports/experiment_summary.json`.
- Learning curves: `results/training_20261031/plots/learning_curves.png`.

The 30-epoch ImageNet-initialized run used all 189 local images: 131 training, 30 validation and 28 test. Epoch 8 was selected using validation macro-F1. Test accuracy was 75.0%, macro-F1 .7056 and macro-AUC .9386. Curves show overfitting; hair corruption reduced accuracy to 46.4%. These are limited local-subset measurements; the full 10,015-image dataset and baseline split provenance are unavailable.

Class order remains `akiec, bcc, bkl, df, mel, nv, vasc`; preprocessing remains RGB 224x224 and original ImageNet normalization. Inference separates original softmax and scaled probabilities, each covering seven classes and summing to approximately one. Evidence lists stay empty because classification does not verify clinical findings. Grad-CAM target metadata remains configured; inference does not compute a heatmap.

## Documentation and archived runs

- [Research pipeline](docs/research_pipeline.md): augmentation, calibration, clinical/statistical methodology and robustness commands.
- [Inference schema](docs/inference_schema.md): exact DiagnosticState keys and probability semantics.
- [Historical smoke results](docs/smoke_results.md): measured integration checks, separate from the trained model.

Earlier complete smoke runs were archived with per-file SHA256 verification. Their manifest is `results/archive/index.json`. Restore a run to its original paths with:

```powershell
Expand-Archive -LiteralPath results/archive/smoke_final.zip -DestinationPath results
```

The baseline, upgraded weights, calibration, dataset and active experiment records are retained. Obsolete schema documentation and generated project bytecode caches were removed. Useful tests and locally required dependencies remain.
