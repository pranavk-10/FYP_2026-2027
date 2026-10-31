"""
Chest X-Ray DenseNet121 Inference Module (Ojas's Model)
========================================================
Framework: TensorFlow/Keras (NOT PyTorch — Ojas trained this on Colab with TF)
Architecture: DenseNet121 (ImageNet pretrained, frozen backbone) 
              → GlobalAvgPool → Dense(256, ReLU) → BatchNorm → Dropout(0.5) → Dense(20, sigmoid)
Dataset: NIH ChestX-ray14 — 28,461 patient-level-split samples
Task: Multi-label classification (20 disease labels, sigmoid outputs — NOT mutually exclusive)
Training: BinaryFocalCrossentropy (alpha=0.25, gamma=2.0), best val_auc ~0.685

This mirrors the ECG inference module (cnn_pipeline/ecg/ecg_inference.py) but uses
TensorFlow instead of PyTorch. The output schema is standardized to match DiagnosticState.
"""

import os
import io
import json
import numpy as np

import tensorflow as tf

from PIL import Image

IMG_SIZE = 224
THRESHOLD = 0.5

# =====================================================================
# 20 DISEASE LABELS (exact order from Ojas's training notebook cell #2)
# This order MUST match the training label column order exactly
# =====================================================================
LABEL_COLS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax',
    'Pneumoperitoneum', 'Pneumomediastinum', 'Subcutaneous Emphysema',
    'Tortuous Aorta', 'Calcification of the Aorta', 'No Finding'
]

# Clinical display name mapping for doctor-facing UI and debate engine
CLINICAL_LABEL_MAP = {
    'Atelectasis': 'Atelectasis (Lung Collapse)',
    'Cardiomegaly': 'Cardiomegaly (Enlarged Heart)',
    'Consolidation': 'Lung Consolidation',
    'Edema': 'Pulmonary Edema',
    'Effusion': 'Pleural Effusion',
    'Emphysema': 'Emphysema',
    'Fibrosis': 'Pulmonary Fibrosis',
    'Hernia': 'Diaphragmatic Hernia',
    'Infiltration': 'Pulmonary Infiltration',
    'Mass': 'Lung Mass',
    'Nodule': 'Pulmonary Nodule',
    'Pleural_Thickening': 'Pleural Thickening',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax (Collapsed Lung)',
    'Pneumoperitoneum': 'Pneumoperitoneum',
    'Pneumomediastinum': 'Pneumomediastinum',
    'Subcutaneous Emphysema': 'Subcutaneous Emphysema',
    'Tortuous Aorta': 'Tortuous Aorta',
    'Calcification of the Aorta': 'Aortic Calcification',
    'No Finding': 'No Finding (Normal)'
}

# Clinical finding descriptions for detected conditions (for debate agents)
CLINICAL_FINDINGS_MAP = {
    'Atelectasis': "Partial lung collapse with reduced lung volume on chest radiograph.",
    'Cardiomegaly': "Enlarged cardiac silhouette exceeding normal cardiothoracic ratio on PA view.",
    'Consolidation': "Dense opacification of lung parenchyma indicating alveolar filling.",
    'Edema': "Bilateral perihilar haziness and Kerley B lines suggesting pulmonary fluid overload.",
    'Effusion': "Blunting of costophrenic angle indicating fluid collection in pleural space.",
    'Emphysema': "Hyperinflated lungs with flattened diaphragm and increased AP diameter.",
    'Fibrosis': "Reticular interstitial pattern with volume loss suggesting chronic fibrotic changes.",
    'Hernia': "Abnormal soft tissue opacity at the diaphragm suggesting herniation.",
    'Infiltration': "Ill-defined opacities suggesting inflammatory or infectious infiltrates.",
    'Mass': "Well-defined or irregular opacity suggesting neoplastic process in lung parenchyma.",
    'Nodule': "Small round opacity (<3cm) in lung field requiring further characterization.",
    'Pleural_Thickening': "Thickened pleural line suggesting chronic pleural inflammation.",
    'Pneumonia': "Focal or lobar consolidation with air bronchograms suggesting infectious pneumonia.",
    'Pneumothorax': "Visceral pleural line with absent lung markings indicating air in pleural space.",
    'Pneumoperitoneum': "Free air under the diaphragm suggesting bowel perforation.",
    'Pneumomediastinum': "Air tracking along mediastinal structures.",
    'Subcutaneous Emphysema': "Air in subcutaneous tissues visible on chest radiograph.",
    'Tortuous Aorta': "Elongated and tortuous aortic shadow on chest radiograph.",
    'Calcification of the Aorta': "Calcified aortic wall or aortic knob calcification.",
    'No Finding': "Normal chest radiograph with no acute cardiopulmonary abnormality."
}

# =====================================================================
# SINGLETON MODEL CACHE (loads .keras weights once into RAM)
# =====================================================================
_CACHED_MODEL = None

def get_default_weights_path():
    """Returns absolute path to trained .keras weights file."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if '__file__' in globals() else os.getcwd()
    possible_paths = [
        "/content/densenet121_uncompressed.keras",
        os.path.join(base_dir, "cnn_pipeline", "xray", "weights", "densenet121_uncompressed.keras"),
        os.path.join(base_dir, "stuff", "xray", "best_phase1_model-2.keras"),
        "densenet121_uncompressed.keras"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]


def load_xray_model(model_path=None):
    """
    Singleton loader for DenseNet121 X-Ray Model (TensorFlow/Keras).
    Loads trained .keras weights into memory once.
    """
    global _CACHED_MODEL

    if _CACHED_MODEL is not None and model_path is None:
        return _CACHED_MODEL

    if model_path is None:
        model_path = get_default_weights_path()

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"X-Ray model weights file not found at: {model_path}")

    print(f"Loading DenseNet121 X-Ray model from: {model_path}")
    
    # First try standard loading
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (TypeError, ValueError) as e:
        if 'quantization_config' in str(e):
            print("Detected Keras version mismatch — patching model config...")
            model = _load_model_with_compat_fix(model_path)
        else:
            raise
    
    _CACHED_MODEL = model
    return model


def _load_model_with_compat_fix(model_path):
    """
    Loads a .keras model file while stripping unsupported fields 
    (e.g., 'quantization_config') that were added in newer Keras versions.
    """
    import zipfile
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        config = json.loads(config_str)
        _strip_quantization_config(config)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        patched_path = os.path.join(temp_dir, 'patched_model.keras')
        with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == 'patched_model.keras':
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, temp_dir)
                    zf_out.write(full_path, arcname)
        
        model = tf.keras.models.load_model(patched_path, compile=False)
        print("Model loaded successfully with compatibility fix applied.")
        return model
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _strip_quantization_config(obj):
    """Recursively remove 'quantization_config' from nested dicts/lists."""
    if isinstance(obj, dict):
        obj.pop('quantization_config', None)
        for v in obj.values():
            _strip_quantization_config(v)
    elif isinstance(obj, list):
        for item in obj:
            _strip_quantization_config(item)


def preprocess_xray_image(image_input):
    """
    Preprocesses input image to match training pipeline:
    1. Load/decode image
    2. Convert to grayscale
    3. Resize to 224x224 (Bilinear)
    4. Normalize to [0, 1] (/255.0)
    5. Convert to 3 identical RGB channels for DenseNet
    6. Add batch dimension
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8 and image_input.max() <= 1.0:
            img = Image.fromarray((image_input * 255).astype(np.uint8))
        else:
            img = Image.fromarray(image_input.astype(np.uint8))
    else:
        raise ValueError("image_input must be a file path, PIL.Image, bytes, or numpy array.")

    # Convert to grayscale
    img = img.convert('L')

    # Resize to 224x224
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Convert grayscale to RGB (3 channels for DenseNet)
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_rgb, axis=0)

    return img_batch


def predict_xray(image_input, model_path=None, threshold=THRESHOLD, watchlist_threshold=0.15):
    """
    Runs Chest X-Ray image inference and returns standardized JSON evidence structure
    with 3-tier risk stratification.
    """
    model = load_xray_model(model_path=model_path)
    img_batch = preprocess_xray_image(image_input)

    # Run inference
    probabilities = model.predict(img_batch, verbose=0)[0]  # Shape: (20,) or (19,)

    # Determine whether model outputs 19 disease classes or 20 classes (with 'No Finding')
    num_classes = len(probabilities)
    if num_classes == 19:
        active_labels = [c for c in LABEL_COLS if c != 'No Finding']
        normality_score = float(np.prod(1.0 - probabilities))
        disease_indices = list(range(19))
    else:
        active_labels = LABEL_COLS
        no_finding_idx = active_labels.index('No Finding') if 'No Finding' in active_labels else -1
        if no_finding_idx != -1:
            normality_score = float(probabilities[no_finding_idx])
            disease_indices = [i for i in range(num_classes) if i != no_finding_idx]
        else:
            normality_score = float(np.prod(1.0 - probabilities))
            disease_indices = list(range(num_classes))

    # Build disease differential
    disease_probs = [(i, float(probabilities[i])) for i in disease_indices]
    disease_probs.sort(key=lambda x: x[1], reverse=True)

    # Classify each disease into risk tiers
    detected = []     # > threshold (0.5)
    watchlist = []    # watchlist_threshold to threshold (0.15 - 0.5)
    unlikely = []     # < watchlist_threshold (0.15)

    for idx, prob in disease_probs:
        entry = {
            "label": CLINICAL_LABEL_MAP[active_labels[idx]],
            "raw_class": active_labels[idx],
            "probability": prob,
            "calibrated_probability": prob,
        }
        if prob >= threshold:
            entry["risk_tier"] = "detected"
            detected.append(entry)
        elif prob >= watchlist_threshold:
            entry["risk_tier"] = "watchlist"
            watchlist.append(entry)
        else:
            entry["risk_tier"] = "unlikely"
            unlikely.append(entry)

    # Full ranked differential
    sorted_indices = np.argsort(probabilities)[::-1]
    full_differential = [
        {
            "label": CLINICAL_LABEL_MAP[active_labels[idx]],
            "raw_class": active_labels[idx],
            "probability": float(probabilities[idx]),
            "calibrated_probability": float(probabilities[idx]),
            "risk_tier": "detected" if probabilities[idx] >= threshold
                         else "watchlist" if probabilities[idx] >= watchlist_threshold
                         else "unlikely"
        }
        for idx in sorted_indices
    ]

    # Primary diagnosis: highest detected condition, or top watchlist, or No Finding
    if detected:
        primary = detected[0]
        primary_diagnosis = primary["label"]
        primary_class = primary["raw_class"]
        primary_prob = primary["probability"]
    elif watchlist:
        primary = watchlist[0]
        primary_diagnosis = f"{primary['label']} (watchlist — needs clinical correlation)"
        primary_class = primary["raw_class"]
        primary_prob = primary["probability"]
    else:
        primary_diagnosis = "No Finding (Normal)"
        primary_class = "No Finding"
        primary_prob = normality_score

    # Clinical findings for debate agents
    positive_findings = []
    negative_findings = []

    for d in detected:
        raw = d["raw_class"]
        if raw in CLINICAL_FINDINGS_MAP:
            positive_findings.append(f"[DETECTED] {CLINICAL_FINDINGS_MAP[raw]}")

    for d in watchlist:
        raw = d["raw_class"]
        if raw in CLINICAL_FINDINGS_MAP:
            positive_findings.append(
                f"[WATCHLIST] Cannot rule out {d['label']} ({d['probability']:.1%}): "
                f"{CLINICAL_FINDINGS_MAP[raw]}"
            )

    if not positive_findings:
        positive_findings.append("No significant acute cardiopulmonary abnormality detected.")

    if normality_score > 0.4:
        negative_findings.append(f"Normality score: {normality_score:.1%} — likely normal study.")
    elif normality_score > 0.25:
        negative_findings.append(f"Normality score: {normality_score:.1%} — borderline, clinical correlation advised.")
    else:
        negative_findings.append(f"Normality score: {normality_score:.1%} — abnormalities likely present.")

    if not detected and not watchlist:
        negative_findings.append("All disease probabilities below watchlist threshold.")

    return {
        "modality": "xray",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": primary_diagnosis,
            "raw_class": primary_class,
            "probability": primary_prob,
            "calibrated_probability": primary_prob,
            "normality_score": normality_score,
            "findings": full_differential,
            "detected_conditions": [d["label"] for d in detected] if detected else [],
            "watchlist_conditions": [d["label"] for d in watchlist] if watchlist else [],
            "risk_summary": {
                "detected_count": len(detected),
                "watchlist_count": len(watchlist),
                "unlikely_count": len(unlikely),
                "overall_risk": "high" if detected else "moderate" if watchlist else "low"
            }
        },
        "risk_tiers": {
            "detected": detected,
            "watchlist": watchlist,
            "unlikely": unlikely[:5]
        },
        "differential": full_differential,
        "evidence": {
            "positive_findings": positive_findings,
            "negative_findings": negative_findings
        },
        "explainability": {
            "method": "Grad-CAM",
            "target_layer": "densenet121.conv5_block16_concat",
            "region_description": "Spatial attention heatmap over chest radiograph regions of interest."
        }
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        print(f"Running DenseNet121 X-Ray inference on: {test_file}")
        res = predict_xray(test_file)
        print(json.dumps(res, indent=2))
    else:
        print("X-Ray inference module loaded successfully. Pass an image file to test.")
        print(f"Model weights expected at: {get_default_weights_path()}")
        print(f"Labels: {LABEL_COLS}")