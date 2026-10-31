import os
import json
import io
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path

if __package__:
    from .core.common import CLASS_INDEX_TO_NAME, IMG_SIZE, load_checkpoint
    from .core.augmentation import get_eval_transform
    from .core.calibration import load_temperature
else:
    from core.common import CLASS_INDEX_TO_NAME, IMG_SIZE, load_checkpoint
    from core.augmentation import get_eval_transform
    from core.calibration import load_temperature

# The original class-index and preprocessing contracts live in common.py.

# Clinical display name mapping for doctor-facing UI and debate engine
CLINICAL_LABEL_MAP = {
    'akiec': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesion',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion'
}

_CACHED_MODEL = None
_CACHED_DEVICE = None
_CACHED_KEY = None

def get_default_weights_path():
    """Returns absolute path to trained weights file for skin lesion model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
        os.path.join(base_dir, "cnn_pipeline", "skin", "weights", "model_ham10000_upgraded.pth"),
        os.path.join(base_dir, "cnn_pipeline", "skin", "weights", "model_ham10000_best.pth"),
        os.path.join(base_dir, "cnn_pipeline", "skin", "model_ham10000_best.pth"),
        "model_ham10000_best.pth"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]


def load_skin_model(model_path=None, device=None):
    """
    Singleton loader for ResNet18 Skin Lesion Model.
    Loads trained weights into memory once.
    """
    global _CACHED_MODEL, _CACHED_DEVICE, _CACHED_KEY

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if model_path is None:
        model_path = get_default_weights_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Skin lesion model weights file not found at: {model_path}")

    path = Path(model_path).resolve()
    calibration_path = os.environ.get('SKIN_CALIBRATION_PATH')
    artifact = Path(calibration_path) if calibration_path else (path.with_name('calibration_temperature.json')
               if path.name == 'model_ham10000_upgraded.pth' else path.with_suffix('.calibration.json'))
    if calibration_path and not artifact.is_file():
        raise FileNotFoundError(f'Explicit calibration artifact not found: {artifact}')
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size, str(device), str(artifact.resolve()),
           artifact.stat().st_mtime_ns if artifact.exists() else None)
    if _CACHED_MODEL is not None and _CACHED_KEY == key:
        return _CACHED_MODEL, _CACHED_DEVICE
    model, _ = load_checkpoint(path, device)
    model._skin_temperature = load_temperature(path, artifact)

    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    _CACHED_KEY = key

    return model, device


def get_skin_transform():
    """Returns PyTorch evaluation image transforms matching training."""
    return get_eval_transform()


def predict_skin(image_input, model_path=None, device=None):
    """
    Runs skin lesion dermoscopy image inference and returns standardized JSON evidence structure.

    Args:
        image_input: Filepath (str/Path), PIL.Image object, or raw image bytes.
        model_path: Optional path to weights file.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict matching standardized DiagnosticState input schema.
    """
    model, device = load_skin_model(model_path=model_path, device=device)
    transform = get_skin_transform()

    # Handle flexible input types (filepath, PIL image, or bytes)
    if isinstance(image_input, (str, bytes, os.PathLike)):
        if isinstance(image_input, bytes):
            img = Image.open(io.BytesIO(image_input)).convert('RGB')
        else:
            img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("image_input must be a file path, PIL.Image, or bytes.")

    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        calibrated_probs = F.softmax(logits / getattr(model, '_skin_temperature', 1.), dim=1).cpu().numpy()[0]

    # Rank differentials by probability descending
    sorted_indices = np.argsort(calibrated_probs)[::-1]

    differential = [
        {
            "label": CLINICAL_LABEL_MAP[CLASS_INDEX_TO_NAME[idx]],
            "raw_class": CLASS_INDEX_TO_NAME[idx],
            "probability": float(probs[idx]),
            "calibrated_probability": float(calibrated_probs[idx])
        }
        for idx in sorted_indices
    ]

    top_idx = sorted_indices[0]
    top_raw = CLASS_INDEX_TO_NAME[top_idx]
    top_clinical = CLINICAL_LABEL_MAP[top_raw]
    top_prob = float(probs[top_idx])

    # A classifier does not establish dermoscopic findings or rule out pathology.
    # Keep evidence fields for DiagnosticState without inventing observations.
    positive_findings = []
    negative_findings = []

    return {
        "modality": "dermoscopy",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": top_clinical,
            "raw_class": top_raw,
            "probability": top_prob,
            "calibrated_probability": float(calibrated_probs[top_idx]),
            "findings": differential
        },
        "differential": differential,
        "evidence": {
            "positive_findings": positive_findings,
            "negative_findings": negative_findings
        },
        "explainability": {
            "method": "Grad-CAM",
            "target_layer": "resnet18.layer4",
            "region_description": "Grad-CAM target is configured; predict_skin did not compute a heatmap or verify lesion localization."
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        res = predict_skin(test_file)
        print(json.dumps(res, indent=2))
    else:
        print("Skin inference module loaded successfully. Pass an image file to test.")
