# Skin lesion (dermoscopy) inference, model v2: same interface as v1, ECG-style output.
#
#     from skin_inference import predict_skin
#     result = predict_skin("lesion.jpg")                 # path, raw bytes or PIL.Image
#     result = predict_skin(img_bytes, gradcam=True)      # adds a heatmap (base64 PNG)
#
# Needs: torch, torchvision, pillow, numpy.
# Weights: skin_v2_deploy.pth (set SKIN_V2_WEIGHTS or put it in ./weights/).
import os
import io
import json
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

WEIGHTS_FILENAME = "skin_v2_deploy.pth"

# Class-level descriptions only: the model scores the whole image and does not detect individual
# dermoscopic structures, so none are named here (agents tended to read them as observed).
CLASS_TEXT = {
    "akiec": "Whole-image pattern classified as actinic keratosis / intraepithelial carcinoma.",
    "bcc":   "Whole-image pattern classified as basal cell carcinoma.",
    "bkl":   "Whole-image pattern classified as a benign keratosis-like lesion.",
    "df":    "Whole-image pattern classified as dermatofibroma.",
    "mel":   "Whole-image pattern classified as melanoma.",
    "nv":    "Whole-image pattern classified as a melanocytic nevus.",
    "vasc":  "Whole-image pattern classified as a vascular lesion.",
}
LIMITATION_NOTE = ("Model covers 7 HAM10000 dermoscopy classes only and scores the whole image; it does not "
                   "assess dermoscopic structures (network, globules, borders), lesion evolution or history, "
                   "and is not validated on non-dermoscopic phone photos.")

_CACHE = {"model": None, "device": None, "cfg": None, "path": None}


def get_default_weights_path():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("SKIN_V2_WEIGHTS", ""), os.path.join(here, "weights", WEIGHTS_FILENAME),
                  os.path.join(here, WEIGHTS_FILENAME), WEIGHTS_FILENAME]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[1]


def _build(cfg):
    from torchvision import models
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(cfg["classes"]))
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in cfg["model_fp16"].items()})
    return model


def load_skin_model(model_path=None, device=None):
    # Loads the model once and keeps it in memory. Returns (model, device).
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_path = model_path or get_default_weights_path()
    if _CACHE["model"] is not None and _CACHE["device"] == device and _CACHE["path"] == model_path:
        return _CACHE["model"], device
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Skin v2 weights not found at: {model_path}")
    cfg = torch.load(model_path, map_location="cpu", weights_only=True)
    model = _build(cfg).to(device).eval()
    _CACHE.update(model=model, device=device, cfg=cfg, path=model_path)
    return model, device


def _to_pil(image_input):
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    if isinstance(image_input, (str, os.PathLike)):
        with Image.open(image_input) as im:
            return im.convert("RGB")
    raise ValueError("image_input must be a file path, PIL.Image, or bytes.")


def preprocess_skin_image(image_input, cfg):
    # RGB, bilinear resize to 224x224, ImageNet normalisation (same as training). Returns (tensor, resized PIL).
    s = cfg["input_size"]
    img = _to_pil(image_input).resize((s, s), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    x = (x - torch.tensor(cfg["mean"]).view(3, 1, 1)) / torch.tensor(cfg["std"]).view(3, 1, 1)
    return x.unsqueeze(0), img


def _gradcam(model, x, class_idx):
    acts = {}
    h = model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
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


def _overlay_png_base64(img, cam):
    heat = np.stack([np.clip(1.5 - np.abs(4 * cam - k), 0, 1) for k in (3, 2, 1)], -1) * 255
    blended = (0.6 * np.asarray(img, dtype=np.float32) + 0.4 * heat).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def predict_skin(image_input, model_path=None, device=None, gradcam=False):
    # Runs dermoscopy inference and returns the standardized JSON evidence packet.
    # Risk tiers per class use calibrated probabilities and thresholds fitted on validation:
    #   detected  : p >= threshold (best one-vs-rest F1)
    #   watchlist : watch threshold <= p < threshold (melanoma/BCC/AKIEC: >= 90% validation sensitivity)
    #   unlikely  : below that
    model, device = load_skin_model(model_path=model_path, device=device)
    cfg = _CACHE["cfg"]
    classes, labels, T = cfg["classes"], cfg["labels"], float(cfg["temperature"])
    th, watch, rules, malignant = cfg["thresholds"], cfg["watch_thresholds"], cfg["rules"], cfg["malignant"]

    x, img = preprocess_skin_image(image_input, cfg)
    x = x.to(device)
    with torch.no_grad():
        logits = model(x).float().cpu()[0]
    raw = torch.softmax(logits, 0).numpy()
    cal = torch.softmax(logits / T, 0).numpy()

    def tier(c, p):
        if p >= th[c]:
            return "detected"
        if p >= watch[c]:
            return "watchlist"
        return "unlikely"

    entries = {}
    for i, c in enumerate(classes):
        entries[c] = {"label": labels[c], "raw_class": c, "probability": round(float(raw[i]), 4),
                      "calibrated_probability": round(float(cal[i]), 4), "threshold": th[c],
                      "watch_threshold": watch[c], "risk_tier": tier(c, float(cal[i])),
                      "malignant": c in malignant}
    differential = sorted(entries.values(), key=lambda e: e["calibrated_probability"], reverse=True)
    top, second = differential[0], differential[1]
    detected = [e for e in differential if e["risk_tier"] == "detected"]
    watchlist = [e for e in differential if e["risk_tier"] == "watchlist"]
    unlikely = [e for e in differential if e["risk_tier"] == "unlikely"]
    malignancy_score = round(float(sum(cal[classes.index(c)] for c in malignant)), 4)

    if top["risk_tier"] == "detected":
        primary_diagnosis = top["label"]
    else:
        primary_diagnosis = f"{top['label']} (below decision threshold - needs clinical correlation)"

    # ---- referral ("abstain") rules ----
    reasons = []
    if top["calibrated_probability"] < rules["min_top_probability"]:
        reasons.append(f"Low confidence: top class {top['label']} at {top['calibrated_probability']:.0%}.")
    elif top["calibrated_probability"] - second["calibrated_probability"] < rules["min_margin"]:
        reasons.append(f"Two classes nearly tied: {top['label']} {top['calibrated_probability']:.0%} vs "
                       f"{second['label']} {second['calibrated_probability']:.0%}.")
    for e in differential:
        if e["malignant"] and e is not top and e["risk_tier"] != "unlikely":
            reasons.append(f"Cannot rule out {e['label']} ({e['calibrated_probability']:.0%}, above its "
                           "high-sensitivity watch threshold).")
    abstain = bool(reasons)
    abstain_reason = " ".join(reasons) or None

    # ---- evidence for the debate agents ----
    positive_findings, negative_findings = [], []
    for e in detected:
        positive_findings.append(f"[DETECTED] {e['label']} ({e['calibrated_probability']:.1%}): {CLASS_TEXT[e['raw_class']]}")
    for e in watchlist:
        positive_findings.append(f"[WATCHLIST] Cannot rule out {e['label']} ({e['calibrated_probability']:.1%}): "
                                 f"{CLASS_TEXT[e['raw_class']]}")
    if not positive_findings:
        positive_findings.append("No lesion class reaches its decision threshold.")
    negative_findings.append(f"Malignancy score (melanoma + BCC + AKIEC): {malignancy_score:.1%}.")
    for e in unlikely:
        negative_findings.append(f"{e['label']} unlikely ({e['calibrated_probability']:.1%}).")
    negative_findings.append(LIMITATION_NOTE)
    if abstain:
        # agents read only the first negative finding, so the referral reason goes first
        negative_findings.insert(0, f"[REFER] Specialist review advised: {abstain_reason}")

    explain = {"method": "Grad-CAM", "target_layer": "resnet18.layer4",
               "target_class": top["raw_class"], "region_description": "Spatial attention over the dermoscopy image."}
    if gradcam:
        cam = _gradcam(model, x, classes.index(top["raw_class"]))
        explain["heatmap_png_base64"] = _overlay_png_base64(img, cam)

    any_malignant = any(e["malignant"] for e in detected + watchlist)
    return {
        "modality": "dermoscopy",
        "input_type": "image",
        "prediction": {
            "primary_diagnosis": primary_diagnosis,
            "raw_class": top["raw_class"],
            "probability": top["probability"],
            "calibrated_probability": top["calibrated_probability"],
            "malignancy_score": malignancy_score,
            "findings": differential,
            "detected_conditions": [e["label"] for e in detected],
            "watchlist_conditions": [e["label"] for e in watchlist],
            "abstain": abstain,
            "abstain_reason": abstain_reason,
            "risk_summary": {"detected_count": len(detected), "watchlist_count": len(watchlist),
                             "unlikely_count": len(unlikely),
                             "overall_risk": "high" if any_malignant else "moderate" if abstain else "low"},
        },
        "risk_tiers": {"detected": detected, "watchlist": watchlist, "unlikely": unlikely},
        "differential": differential,
        "evidence": {"positive_findings": positive_findings, "negative_findings": negative_findings},
        "explainability": explain,
        "model_info": {"version": f"skin-v2-{cfg['arch']}", "classes": classes, "temperature": round(T, 4),
                       "thresholds": th, "watch_thresholds": watch, "test_metrics": cfg["test_metrics"],
                       "trained_on": cfg["trained_on"]},
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(predict_skin(sys.argv[1]), indent=2))
    else:
        print("Skin v2 inference module loaded. Pass an image file to test.")
