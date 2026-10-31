# Measured integration smoke test

These are execution checks, **not research or clinical performance claims**. Only 189/10,015 dataset images were available. The controlled subset contained 58 real images from 56 lesions (eight lesions per class). Splits contained 29 training images/28 lesions, 14 validation images/14 lesions, and 15 test images/14 lesions. No lesion crossed a split.

The upgraded model trained for one CPU epoch with random initialization, seed 123, batch size 4 and custom augmentation. The comparison used the existing legacy checkpoint on the same 15 ordered test images. Its training split is unknown and may overlap these images; the comparison cannot establish unbiased generalization or improvement.

| Test metric | Existing baseline | One-epoch upgraded smoke model |
|---|---:|---:|
| Accuracy | 0.400000 | 0.133333 |
| Macro-F1 | 0.328571 | 0.040816 |
| Macro ROC-AUC | 0.832723 | 0.503663 |
| Macro sensitivity | 0.404762 | 0.142857 |
| Macro specificity | 0.901099 | 0.857143 |
| ECE (10 bins) | 0.271492 | 0.039806 |
| Brier score (sum over classes) | 0.746911 | 0.858888 |

Baseline probabilities are original softmax (T=1). Upgraded probabilities use validation-fitted T=2.238698. Validation NLL changed from 1.950763 to 1.935939; validation ECE changed from .078499 to .032719. Low ECE here does not imply a useful classifier: the upgraded model predicts a nearly uniform distribution and performs poorly. The comparison report's supported accuracy-improvement flag is false.

All ten corruptions ran at severity 1. Clean accuracy was .133333; brightness corruption accuracy was .066667, and the other nine conditions had .133333 accuracy. FGSM and two-iteration PGD at epsilon 2/255 had accuracy 0. These are measurements of the undertrained smoke model only. Detailed per-class, AUC, calibration and degradation values are in the saved robustness report.

Paired McNemar (exact p=.125), 30-draw lesion bootstrap, ROC/PR/reliability/decision curves, checkpoint reload and DiagnosticState inference completed. Eleven mathematical/contract tests passed. A repeat run produced identical split assignments and identical model parameter/buffer hashes; robustness confirmed unchanged model parameters and buffers. Baseline SHA256 before and after was:

```text
558624539632d95c6d6507459ff136ee520662b9b1079e439bf8bf675acbb3f2
```

Both complete runs are archived in `results/archive/smoke_final.zip` and `results/archive/smoke_verified.zip`. Original paths and file SHA256 values are recorded in `results/archive/index.json`. Extract an archive into `results/` to restore its original directory, including data, weights, metrics, plots and configuration. The active 30-epoch run remains in `results/training_20261031/`; active weights remain in `weights/`.

The tested environment was the project's `.venv` Python 3.13.15 with CPU torch 2.14.1, torchvision .29.1, and workspace-local analysis packages. To repeat in that environment from `cnn_pipeline/skin/` (use a fresh results directory):

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) '.research-deps'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& ../../.venv/Scripts/python.exe -m unittest discover -s tests -p test_research_pipeline.py -v
& ../../.venv/Scripts/python.exe tests/smoke_experiment.py --results-dir results/smoke_repeat
```

In a clean environment, install `requirements.txt` and use ordinary `python` commands as documented in the root README. Full training, at least 1000 bootstrap draws and provenance-verified paired baseline training remain necessary for a research comparison.
