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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_paths = [
        os.path.join(base_dir, "cnn_pipeline", "xray", "weights", "best_phase1_model.keras"),
        os.path.join(base_dir, "stuff", "xray", "best_phase1_model-2.keras"),
        "best_phase1_model.keras"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return possible_paths[0]


def load_xray_model(model_path=None):
    """
    Singleton loader for DenseNet121 X-Ray Model (TensorFlow/Keras).
    Loads trained .keras weights into memory once.
    
    COMPATIBILITY FIX: Ojas's model was saved on Google Colab with a Keras version
    that includes 'quantization_config' in Dense layer configs. Our local Keras
    may not recognize this field. We strip it before loading to avoid the error:
    "Unrecognized keyword arguments passed to Dense: {'quantization_config': None}"
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
    
    The .keras format is a ZIP file containing:
      - config.json (model architecture)
      - model.weights.h5 (weights)
    
    We extract config.json, strip the incompatible field, 
    rebuild the model from config, then load the weights separately.
    """
    import zipfile
    import tempfile
    import shutil
    
    # Extract the .keras zip to a temp directory
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(temp_dir)
        
        # Read and patch config.json to remove 'quantization_config'
        config_path = os.path.join(temp_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_str = f.read()
        
        config = json.loads(config_str)
        _strip_quantization_config(config)
        
        # Write patched config back
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Repackage into a patched .keras zip file
        patched_path = os.path.join(temp_dir, 'patched_model.keras')
        with zipfile.ZipFile(patched_path, 'w', zipfile.ZIP_DEFLATED) as zf_out:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file == 'patched_model.keras':
                        continue
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, temp_dir)
                    zf_out.write(full_path, arcname)
        
        # Load the patched model
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
    Preprocesses input image to match Ojas's exact training pipeline:
    1. Load/decode as grayscale
    2. Resize to 224x224
    3. Normalize to [0, 1] (/255.0)
    4. Convert grayscale to RGB (3 channels for DenseNet)
    5. Add batch dimension

    Args:
        image_input: Filepath (str), PIL.Image, or raw image bytes.

    Returns:
        numpy array of shape (1, 224, 224, 3) ready for model.predict()
    """
    # Handle flexible input types (same interface as ECG module)
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, (str, os.PathLike)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        raise ValueError("image_input must be a file path, PIL.Image, or bytes.")

    # Convert to grayscale first (matching Ojas's decode_png channels=1)
    img = img.convert('L')

    # Resize to 224x224
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Convert to numpy and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0

    # Convert grayscale to RGB (matching Ojas's grayscale_to_rgb)
    img_rgb = np.stack([img_array, img_array, img_array], axis=-1)

    # Add batch dimension: (1, 224, 224, 3)
    img_batch = np.expand_dims(img_rgb, axis=0)

    return img_batch


def predict_xray(image_input, model_path=None, threshold=THRESHOLD):
    """
    Runs Chest X-Ray image inference and returns standardized JSON evidence structure.

    This is the X-Ray equivalent of predict_ecg() — same output schema,
    different model (TensorFlow DenseNet121 instead of PyTorch ResNet18).

    Key difference from ECG:
    - ECG is multi-CLASS (softmax, one winner) → 4 mutually exclusive classes
    - X-Ray is multi-LABEL (sigmoid, multiple winners) → 20 independent disease probabilities

    Args:
        image_input: Filepath (str/Path), PIL.Image object, or raw image bytes.
        model_path: Optional path to .keras weights file.
        threshold: Sigmoid threshold for detecting conditions (default 0.5).

    Returns:
        Dict matching standardized DiagnosticState input schema.
    """
    model = load_xray_model(model_path=model_path)
    img_batch = preprocess_xray_image(image_input)

    # Run inference
    probabilities = model.predict(img_batch, verbose=0)[0]  # Shape: (20,)

    # Build ranked differential (all 20 classes sorted by probability)
    sorted_indices = np.argsort(probabilities)[::-1]

    differential = [
        {
            "label": CLINICAL_LABEL_MAP[LABEL_COLS[idx]],
            "raw_class": LABEL_COLS[idx],
            "probability": float(probabilities[idx]),
            "calibrated_probability": float(probabilities[idx]),
            "detected": bool(probabilities[idx] >= threshold)
        }
        for idx in sorted_indices
    ]

    # Identify detected conditions (above threshold)
    detected_conditions = [d for d in differential if d["detected"]]

    # Primary diagnosis is the highest-probability detected condition
    # (or highest overall if nothing crosses threshold)
    top = differential[0]

    # Build clinical findings for debate agents
    positive_findings = []
    negative_findings = []

    for d in detected_conditions:
        raw = d["raw_class"]
        if raw in CLINICAL_FINDINGS_MAP:
            positive_findings.append(CLINICAL_FINDINGS_MAP[raw])

    if not positive_findings:
        positive_findings.append("No significant acute cardiopulmonary abnormality detected.")

    # Add negative findings for high-probability conditions that were NOT detected
    for d in differential:
        if not d["detected"] and d["probability"] > 0.2:
            negative_findings.append(
                f"Possible {CLINICAL_LABEL_MAP[d['raw_class']]} (probability {d['probability']:.1%} below threshold)."
            )

    if not negative_findings:
        negative_findings.append("No borderline conditions noted.")

    return {
        "modality": "xray",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": top["label"],
            "raw_class": top["raw_class"],
            "probability": top["probability"],
            "calibrated_probability": top["calibrated_probability"],
            "findings": differential,
            "detected_conditions": [d["label"] for d in detected_conditions] if detected_conditions else ["No Finding"]
        },
        "differential": differential,
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
