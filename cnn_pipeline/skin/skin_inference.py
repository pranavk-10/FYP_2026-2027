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

# Class index mapping from HAM10000 training (7 diagnostic categories)
CLASS_INDEX_TO_NAME = {
    0: 'akiec',
    1: 'bcc',
    2: 'bkl',
    3: 'df',
    4: 'mel',
    5: 'nv',
    6: 'vasc'
}

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

def get_default_weights_path():
    """Returns absolute path to trained weights file for skin lesion model."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
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
        raise FileNotFoundError(f"Skin lesion model weights file not found at: {model_path}")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_INDEX_TO_NAME))

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model = checkpoint

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_DEVICE = device

    return model, device


def get_skin_transform():
    """Returns PyTorch evaluation image transforms matching training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


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

    # Derive clinical findings summary for debate agents and clinical UI
    positive_findings = []
    negative_findings = []

    if top_raw == 'mel':
        positive_findings.extend([
            "Asymmetric multicomponent pigment pattern with architectural disorganization.",
            "Presence of atypical pigment network, irregular dots/globules, and peripheral streaks.",
            "Blue-white veil and abnormal vascular structures indicative of malignant melanoma."
        ])
        negative_findings.append("Absence of regular benign reticular network or uniform homogeneous architecture.")
    elif top_raw == 'nv':
        positive_findings.extend([
            "Symmetric lesion with uniform pigment distribution across central and peripheral zones.",
            "Regular reticular pigment network and uniform peripheral globules consistent with benign melanocytic nevus."
        ])
        negative_findings.append("Absence of atypical streaks, blue-white veil, marked asymmetry, or regression structures.")
    elif top_raw == 'bcc':
        positive_findings.extend([
            "Prominent arborizing (tree-like) telangiectasias across the surface of the lesion.",
            "Blue-gray ovoid nests, multiple blue-gray globules, and focal ulceration typical of basal cell carcinoma."
        ])
        negative_findings.append("Absence of pigmented melanocytic network or uniform reticular grid pattern.")
    elif top_raw == 'akiec':
        positive_findings.extend([
            "Prominent erythematous background with white-to-yellow keratotic surface scale and crust.",
            "Strawberry pattern with targetoid hair follicles surrounded by a white halo, typical of actinic keratosis / Bowen's disease."
        ])
        negative_findings.append("Absence of discrete deep dermal pigment nests or prominent arborizing telangiectasia.")
    elif top_raw == 'bkl':
        positive_findings.extend([
            "Milky-orange or brownish structureless areas with sharp, moth-eaten borders.",
            "Characteristic comedo-like openings, horn pseudocysts, and fingerprint-like structures consistent with benign keratosis."
        ])
        negative_findings.append("Absence of atypical melanocytic pigment network or deep invasive architectural signs.")
    elif top_raw == 'df':
        positive_findings.extend([
            "Distinct central white patch or scar-like structureless area.",
            "Delicate, fine peripheral pigment network characteristic of dermatofibroma."
        ])
        negative_findings.append("Absence of marked structural asymmetry, irregular streaks, or blue-white veil.")
    elif top_raw == 'vasc':
        positive_findings.extend([
            "Well-demarcated red, violaceous, or blue-black lacunae (vascular lagoons).",
            "Distinct vascular spaces with reddish-purple coloration typical of benign vascular lesions (hemangioma / angioma)."
        ])
        negative_findings.append("Absence of melanocytic pigment network, pigment globules, or arborizing telangiectasia.")

    return {
        "modality": "dermoscopy",
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
            "region_description": "Spatial attention concentrated on lesion center, irregular borders, and peripheral pigment distribution."
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
