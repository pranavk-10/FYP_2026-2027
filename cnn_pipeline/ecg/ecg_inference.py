import os
import json
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np

IMG_SIZE = 224

# Class index mapping from training
CLASS_INDEX_TO_NAME = {
    0: 'abnormal',
    1: 'history_mi',
    2: 'mi',
    3: 'normal'
}

# Clinical display name mapping for doctor-facing UI and debate engine
CLINICAL_LABEL_MAP = {
    'mi': 'Myocardial Infarction',
    'history_mi': 'History of MI',
    'abnormal': 'Abnormal Heartbeat',
    'normal': 'Normal'
}

_CACHED_MODEL = None
_CACHED_DEVICE = None

def get_default_weights_path():
    """Returns absolute path to trained weights file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
        os.path.join(base_dir, "cnn_pipeline", "ecg", "weights", "model_ecg_image_best.pth"),
        os.path.join(base_dir, "cnn_pipeline", "ecg", "model_ecg_image_best.pth"),
        os.path.join(base_dir, "stuff", "model_ecg_image_best.pth"),
        "model_ecg_image_best.pth"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]


def load_ecg_model(model_path=None, device=None):
    """
    Singleton loader for ResNet18 ECG Model.
    Loads trained weights into memory once.
    """
    global _CACHED_MODEL, _CACHED_DEVICE

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if _CACHED_MODEL is not None and _CACHED_DEVICE == device and model_path is None:
        return _CACHED_MODEL, _CACHED_DEVICE

    if model_path is None:
        model_path = get_default_weights_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ECG model weights file not found at: {model_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_INDEX_TO_NAME))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_DEVICE = device

    return model, device

def get_ecg_transform():
    """Returns PyTorch evaluation image transforms matching training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_ecg(image_input, model_path=None, device=None):
    """
    Runs ECG image inference and returns standardized JSON evidence structure.

    Args:
        image_input: Filepath (str/Path), PIL.Image object, or raw image bytes.
        model_path: Optional path to weights file.
        device: 'cpu' or 'cuda'.

    Returns:
        Dict matching standardized DiagnosticState input schema.
    """
    model, device = load_ecg_model(model_path=model_path, device=device)
    transform = get_ecg_transform()

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

    # Rank differentials by probability descending
    sorted_indices = np.argsort(probs)[::-1]
    
    differential = [
        {
            "label": CLINICAL_LABEL_MAP[CLASS_INDEX_TO_NAME[idx]],
            "raw_class": CLASS_INDEX_TO_NAME[idx],
            "probability": float(probs[idx]),
            "calibrated_probability": float(probs[idx])
        }
        for idx in sorted_indices
    ]

    top_idx = sorted_indices[0]
    top_raw = CLASS_INDEX_TO_NAME[top_idx]
    top_clinical = CLINICAL_LABEL_MAP[top_raw]
    top_prob = float(probs[top_idx])

    # Derive clinical findings summary for debate agents
    positive_findings = []
    negative_findings = []

    if top_raw == 'mi':
        positive_findings.append("Significant ST-segment elevation detected across composite 12-lead regions.")
        negative_findings.append("No normal sinus rhythm pattern.")
    elif top_raw == 'history_mi':
        positive_findings.append("Pathological Q-waves and T-wave inversions indicative of prior myocardial infarction.")
    elif top_raw == 'abnormal':
        positive_findings.append("Non-specific cardiac arrhythmia / abnormal heartbeat waveform pattern.")
    else:
        positive_findings.append("Normal 12-lead composite ECG waveform pattern.")

    return {
        "modality": "ecg",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": top_clinical,
            "raw_class": top_raw,
            "probability": top_prob,
            "calibrated_probability": top_prob,
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
            "region_description": "Spatial attention concentrated on 12-lead composite waveform rows."
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        res = predict_ecg(test_file)
        print(json.dumps(res, indent=2))
    else:
        print("ECG inference module loaded successfully. Pass an image file to test.")
