# Skin inference output contract

Entry points are unchanged:

```python
from cnn_pipeline.skin import predict_skin, load_skin_model
result = predict_skin('patient_lesion.jpg', device='cpu')
```

`predict_skin(image_input, model_path=None, device=None)` accepts a path, PIL image or bytes. `load_skin_model(model_path=None, device=None)` returns `(model, device)`. Default loading prefers upgraded weights and falls back to the baseline. Raw seven-class state dictionaries and structured checkpoints remain supported.

| Field | Type and meaning |
|---|---|
| `modality` | `"dermoscopy"` |
| `input_type` | `"image"` |
| `prediction.primary_diagnosis` | Display label of the highest-ranked class |
| `prediction.raw_class` | Diagnostic code of that class |
| `prediction.probability` | Original softmax probability of the top class |
| `prediction.calibrated_probability` | Temperature-scaled probability of the top class |
| `prediction.findings` | Same seven-item list as `differential` |
| `differential` | All seven diagnoses, sorted by scaled probability |
| `evidence.positive_findings` | Empty list; no dermoscopic findings are verified |
| `evidence.negative_findings` | Empty list; absence of pathology is not established |
| `explainability.method` | `"Grad-CAM"`, retained as configured method metadata |
| `explainability.target_layer` | `"resnet18.layer4"` |
| `explainability.region_description` | States that no heatmap/localization was computed |

Differential items retain `label`, `raw_class`, `probability`, and `calibrated_probability`. Both distributions sum to approximately one. Class indices remain `0 akiec`, `1 bcc`, `2 bkl`, `3 df`, `4 mel`, `5 nv`, `6 vasc`.

Original probability is `softmax(logits)`; scaled probability is `softmax(logits/T)`. The conventional upgraded checkpoint reads adjacent `calibration_temperature.json`; other filenames read `<checkpoint-stem>.calibration.json`. Set `SKIN_CALIBRATION_PATH` to choose an explicit artifact. Artifacts are bound to checkpoint SHA256 and class ordering; mismatches raise errors.

Without a default artifact, T=1 and both fields coincide as uncalibrated softmax. Explicitly missing artifacts raise errors. Positive scalar T preserves class ranking. Scaled probability does not establish verified individual clinical risk.

Preprocessing remains RGB, bilinear resize to 224x224, mean `[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. External attribution code can still target `layer4`; classifier output is not a verified clinical feature description.
