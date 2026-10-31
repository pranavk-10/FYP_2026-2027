"""
ECG image inference, model v3 (ConvNeXt-Tiny, 5 PTB-XL superclasses, multi-label).

Drop-in replacement for the v1 module: same function names and the same output keys,
plus the multi-label fields used by the X-ray module (normality_score, risk tiers).

    from ecg_inference import predict_ecg
    result = predict_ecg("ecg.jpg")                 # path, raw bytes or PIL.Image
    result = predict_ecg(img_bytes, gradcam=True)   # adds a heatmap (base64 PNG)

Needs: torch, timm, pillow, numpy.
Weights: ecg_convnext_v3_deploy.pth (set ECG_V3_WEIGHTS or put it in ./weights/).
"""
import os
import io
import json
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_VERSION = "ecg-v3-convnext-tiny"
WEIGHTS_FILENAME = "ecg_convnext_v3_deploy.pth"

CLINICAL_LABEL_MAP = {
    "NORM": "Normal ECG",
    "MI":   "Myocardial Infarction",
    "STTC": "ST/T-wave Changes",
    "CD":   "Conduction Disturbance",
    "HYP":  "Ventricular Hypertrophy",
}

# Class-level descriptions only. The model scores the whole printout and does not measure individual
# waves or leads, so no wave-level features are named here (agents tended to read them as observed).
CLINICAL_FINDINGS_MAP = {
    "NORM": "Whole-image pattern classified as a normal 12-lead ECG. Individual waves were not measured.",
    "MI":   "Whole-image pattern classified as myocardial infarction; the model cannot tell acute from old MI. "
            "Individual waves were not measured.",
    "STTC": "Whole-image pattern classified as ST/T-wave change (cause not determined by the model). "
            "Individual waves were not measured.",
    "CD":   "Whole-image pattern classified as conduction disturbance (type not determined by the model). "
            "Individual waves were not measured.",
    "HYP":  "Whole-image pattern classified as ventricular hypertrophy. Individual waves were not measured.",
}

LIMITATION_NOTE = ("Model covers 5 PTB-XL diagnostic superclasses only and scores whole-image patterns; it does "
                   "not measure individual waves or leads, assess rhythm (e.g. atrial fibrillation) or "
                   "distinguish acute from old MI.")

_CACHE = {"model": None, "device": None, "cfg": None, "path": None}


# ----------------------------------------------------------------------------- loading
def get_default_weights_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("ECG_V3_WEIGHTS", ""),
        os.path.join(here, "weights", WEIGHTS_FILENAME),
        os.path.join(here, WEIGHTS_FILENAME),
        WEIGHTS_FILENAME,
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[1]


def load_ecg_model(model_path=None, device=None):
    """Loads the model once and keeps it in memory. Returns (model, device)."""
    import timm
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = model_path or get_default_weights_path()
    if _CACHE["model"] is not None and _CACHE["device"] == device and _CACHE["path"] == model_path:
        return _CACHE["model"], device
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ECG v3 weights not found at: {model_path}")

    cfg = torch.load(model_path, map_location="cpu", weights_only=False)
    model = timm.create_model(cfg["timm_name"], pretrained=False, num_classes=len(cfg["classes"]))
    model.load_state_dict({k: v.float() for k, v in cfg["model_fp16"].items()})
    model.to(device).eval()
    _CACHE.update(model=model, device=device, cfg=cfg, path=model_path)
    return model, device


# ----------------------------------------------------------------------------- preprocessing
def _to_pil(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, (str, os.PathLike)):
        return Image.open(image_input).convert("RGB")
    raise ValueError("image_input must be a file path, PIL.Image, or bytes.")


def preprocess_ecg_image(image_input, cfg):
    """Letterbox to the training size (keep aspect ratio, pad), normalise. Returns (tensor, letterboxed PIL)."""
    img = _to_pil(image_input)
    W, H = cfg["input_wh"]
    s = min(W / img.width, H / img.height)
    nw, nh = round(img.width * s), round(img.height * s)
    canvas = Image.new("RGB", (W, H), tuple(cfg["letterbox_fill"]))
    canvas.paste(img.resize((nw, nh), Image.LANCZOS), ((W - nw) // 2, (H - nh) // 2))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(cfg["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["std"]).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), canvas


# ----------------------------------------------------------------------------- grad-cam
def _gradcam(model, x, class_idx):
    acts = {}
    h = model.stages[-1].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    try:
        with torch.enable_grad():
            out = model(x)
            a = acts["a"]
            g = torch.autograd.grad(out[0, class_idx], a)[0]
    finally:
        h.remove()
    cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def _overlay_png_base64(canvas, cam):
    # jet colour map without matplotlib
    heat = np.stack([np.clip(1.5 - np.abs(4 * cam - k), 0, 1) for k in (3, 2, 1)], -1) * 255
    blended = (0.6 * np.asarray(canvas, dtype=np.float32) + 0.4 * heat).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------- main entry
def predict_ecg(image_input, model_path=None, device=None, gradcam=False):
    """
    Runs ECG image inference and returns the standardized JSON evidence packet.

    Risk tiers per condition use calibrated probabilities and per-class thresholds
    tuned on the validation fold:
      detected  : p >= threshold
      watchlist : max(0.15, threshold/2) <= p < threshold
      unlikely  : below that
    """
    model, device = load_ecg_model(model_path=model_path, device=device)
    cfg = _CACHE["cfg"]
    classes, T, th = cfg["classes"], float(cfg["temperature"]), cfg["thresholds"]

    x, canvas = preprocess_ecg_image(image_input, cfg)
    x = x.to(device).contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        logits = model(x).float().cpu()[0]
    raw = torch.sigmoid(logits).numpy()
    cal = torch.sigmoid(logits / T).numpy()

    def tier(c, p):
        if p >= th[c]:
            return "detected"
        if p >= max(0.15, th[c] / 2):
            return "watchlist"
        return "unlikely"

    entries = {}
    for i, c in enumerate(classes):
        entries[c] = {
            "label": CLINICAL_LABEL_MAP[c],
            "raw_class": c,
            "probability": round(float(raw[i]), 4),
            "calibrated_probability": round(float(cal[i]), 4),
            "threshold": th[c],
            "risk_tier": tier(c, float(cal[i])),
        }
    full_differential = sorted(entries.values(), key=lambda e: e["calibrated_probability"], reverse=True)
    disease = [e for e in full_differential if e["raw_class"] != "NORM"]
    detected = [e for e in disease if e["risk_tier"] == "detected"]
    watchlist = [e for e in disease if e["risk_tier"] == "watchlist"]
    unlikely = [e for e in disease if e["risk_tier"] == "unlikely"]
    norm = entries["NORM"]
    normality_score = norm["calibrated_probability"]
    norm_positive = normality_score >= th["NORM"]

    # ---- primary diagnosis + abstain ("refer to specialist") ----
    abstain, abstain_reason = False, None
    if detected:
        primary = detected[0]
        primary_diagnosis, primary_class = primary["label"], primary["raw_class"]
        if norm_positive:
            abstain, abstain_reason = True, "Model rates the ECG as both normal and abnormal (conflicting outputs)."
    elif watchlist:
        primary = watchlist[0]
        primary_diagnosis = f"{primary['label']} (watchlist - needs clinical correlation)"
        primary_class = primary["raw_class"]
        if norm_positive:
            abstain, abstain_reason = True, (f"Borderline: model leans normal ({normality_score:.0%}) but cannot "
                                             f"rule out {primary['label']} ({primary['calibrated_probability']:.0%}).")
        else:
            abstain, abstain_reason = True, "No condition reaches its threshold and normality is low."
    elif norm_positive:
        primary = norm
        primary_diagnosis, primary_class = norm["label"], "NORM"
    else:
        primary = norm
        primary_diagnosis, primary_class = "Indeterminate - refer to specialist", "INDETERMINATE"
        abstain, abstain_reason = True, "No class reaches its threshold, including normal."

    # ---- evidence for the debate agents ----
    positive_findings, negative_findings = [], []
    for e in detected:
        positive_findings.append(f"[DETECTED] {e['label']} ({e['calibrated_probability']:.1%}): "
                                 f"{CLINICAL_FINDINGS_MAP[e['raw_class']]}")
    for e in watchlist:
        positive_findings.append(f"[WATCHLIST] Cannot rule out {e['label']} ({e['calibrated_probability']:.1%}): "
                                 f"{CLINICAL_FINDINGS_MAP[e['raw_class']]}")
    if not positive_findings:
        positive_findings.append(CLINICAL_FINDINGS_MAP["NORM"] if norm_positive
                                 else "No specific abnormality pattern reaches its decision threshold.")
    if normality_score >= th["NORM"]:
        negative_findings.append(f"Normality score: {normality_score:.1%} - likely normal ECG.")
    elif normality_score >= 0.25:
        negative_findings.append(f"Normality score: {normality_score:.1%} - borderline, clinical correlation advised.")
    else:
        negative_findings.append(f"Normality score: {normality_score:.1%} - abnormalities likely present.")
    for e in unlikely:
        negative_findings.append(f"{e['label']} unlikely ({e['calibrated_probability']:.1%}).")
    negative_findings.append(LIMITATION_NOTE)
    if abstain:
        # agents read only the first negative finding, so the referral reason goes first
        negative_findings.insert(0, f"[REFER] Low-confidence result, specialist review advised: {abstain_reason}")

    explain = {
        "method": "Grad-CAM",
        "target_layer": "convnext_tiny.stages[3]",
        "target_class": primary_class if primary_class in classes else "NORM",
        "region_description": "Spatial attention over the 12-lead waveform rows of the printout.",
    }
    if gradcam:
        cam = _gradcam(model, x, classes.index(explain["target_class"]))
        explain["heatmap_png_base64"] = _overlay_png_base64(canvas, cam)

    return {
        "modality": "ecg",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": primary_diagnosis,
            "raw_class": primary_class,
            "probability": primary["probability"],
            "calibrated_probability": primary["calibrated_probability"],
            "normality_score": normality_score,
            "findings": full_differential,
            "detected_conditions": [e["label"] for e in detected],
            "watchlist_conditions": [e["label"] for e in watchlist],
            "abstain": abstain,
            "abstain_reason": abstain_reason,
            "risk_summary": {
                "detected_count": len(detected),
                "watchlist_count": len(watchlist),
                "unlikely_count": len(unlikely),
                "overall_risk": "high" if detected else "moderate" if watchlist or abstain else "low",
            },
        },
        "risk_tiers": {"detected": detected, "watchlist": watchlist, "unlikely": unlikely},
        "differential": full_differential,
        "evidence": {"positive_findings": positive_findings, "negative_findings": negative_findings},
        "explainability": explain,
        "model_info": {
            "version": MODEL_VERSION,
            "classes": classes,
            "temperature": round(T, 4),
            "thresholds": th,
            "test_macro_auroc": cfg.get("test_macro_auroc"),
            "trained_on": cfg.get("trained_on"),
        },
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(predict_ecg(sys.argv[1]), indent=2))
    else:
        print("ECG v3 inference module loaded. Pass an image file to test.")
