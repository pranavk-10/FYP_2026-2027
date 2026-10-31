"""Shared calibration artifacts, validation-only temperature fitting and diagnostics."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from .common import CLASS_NAMES, file_sha256, save_json


def validate_probabilities(labels, probabilities):
    labels, p = np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or p.shape != (len(labels), len(CLASS_NAMES)) or len(labels) == 0:
        raise ValueError('Expected non-empty labels (N,) and probabilities (N,7).')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Probabilities must be finite, nonnegative and sum to one.')
    if (labels < 0).any() or (labels >= len(CLASS_NAMES)).any():
        raise ValueError('Labels must use the original class indices 0..6.')
    return labels, p


def probabilities_from_logits(logits, temperature=1.):
    logits = np.asarray(logits, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0 or logits.ndim != 2 or logits.shape[1] != 7 or not np.isfinite(logits).all():
        raise ValueError('Expected finite (N,7) logits and a finite positive temperature.')
    return softmax(logits / temperature, axis=1)


def reliability_bins(confidence, correctness, n_bins=15):
    """Equal-width bins: [0,1/B], (1/B,2/B], ..., ((B-1)/B,1]. Empty bins contribute zero."""
    if n_bins < 1:
        raise ValueError('n_bins must be positive.')
    confidence, correctness = np.asarray(confidence), np.asarray(correctness)
    indices = np.maximum(0, np.searchsorted(np.linspace(0, 1, n_bins + 1), confidence, side='left') - 1)
    rows = []
    for b in range(n_bins):
        mask = indices == b
        count = int(mask.sum())
        rows.append({'lower': b / n_bins, 'upper': (b + 1) / n_bins, 'count': count,
                     'confidence': float(confidence[mask].mean()) if count else None,
                     'accuracy': float(correctness[mask].mean()) if count else None})
    return rows


def calibration_metrics(labels, probabilities, n_bins=15):
    labels, p = validate_probabilities(labels, probabilities)
    bins = reliability_bins(p.max(1), p.argmax(1) == labels, n_bins)
    occupied = [b for b in bins if b['count']]
    gaps = [abs(b['accuracy'] - b['confidence']) for b in occupied]
    one_hot = np.eye(7)[labels]
    classwise = {}
    for c, name in enumerate(CLASS_NAMES):
        rows = reliability_bins(p[:, c], labels == c, n_bins)
        ece = sum(b['count'] / len(labels) * abs(b['accuracy'] - b['confidence']) for b in rows if b['count'])
        classwise[name] = {'ece': ece, 'bins': rows}
    return {'ece': sum(b['count'] / len(labels) * gap for b, gap in zip(occupied, gaps)),
            'mce': max(gaps, default=0.), 'brier_score': float(np.mean(np.sum((p - one_hot)**2, axis=1))),
            'nll': float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
            'n_bins': n_bins, 'binning': '[0,1/B], then left-open/right-closed equal-width bins; empty bins ignored',
            'bins': bins, 'classwise': classwise}


def fit_temperature(logits, labels, split='validation'):
    if split != 'validation':
        raise ValueError('Temperature fitting is permitted on validation data only.')
    logits = np.asarray(logits, dtype=float)
    labels, _ = validate_probabilities(labels, probabilities_from_logits(logits))

    def objective(log_temperature):
        from scipy.special import logsumexp
        z = logits / np.exp(log_temperature)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]))

    result = minimize_scalar(objective, bounds=(-3., 3.), method='bounded', options={'xatol': 1e-7})
    if not result.success:
        raise RuntimeError(f'Temperature optimization failed: {result.message}')
    temperature = float(np.exp(result.x)) if result.fun <= objective(0.) else 1.
    return temperature, {'method': 'bounded scalar minimization of unweighted validation NLL',
                          'temperature_bounds': [float(np.exp(-3)), float(np.exp(3))],
                          'at_bound': bool(abs(abs(result.x) - 3) < 1e-4)}


def save_temperature(path, temperature, checkpoint_path, split_path, extra=None):
    save_json(path, {'temperature': temperature, 'fit_split': 'validation', 'class_names': CLASS_NAMES,
                     'checkpoint_sha256': file_sha256(checkpoint_path), 'split_sha256': file_sha256(split_path),
                     **(extra or {})})


def load_temperature(checkpoint_path, artifact_path=None, split_path=None):
    checkpoint_path = Path(checkpoint_path)
    if artifact_path is None:
        # The conventional upgraded name gets the conventional artifact; legacy weights
        # only get their own sidecar, never another model's temperature.
        artifact_path = checkpoint_path.with_name('calibration_temperature.json') if checkpoint_path.name == 'model_ham10000_upgraded.pth' else checkpoint_path.with_suffix('.calibration.json')
    artifact_path = Path(artifact_path)
    if not artifact_path.exists():
        if split_path is not None:
            raise FileNotFoundError(f'Calibration artifact missing: {artifact_path}')
        return 1.
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    if artifact.get('fit_split') != 'validation' or artifact.get('class_names') != CLASS_NAMES:
        raise ValueError('Invalid calibration artifact provenance/class mapping.')
    if artifact.get('checkpoint_sha256') != file_sha256(checkpoint_path):
        raise ValueError('Calibration artifact belongs to a different checkpoint.')
    if split_path is not None and artifact.get('split_sha256') != file_sha256(split_path):
        raise ValueError('Calibration artifact belongs to a different split manifest.')
    t = float(artifact['temperature'])
    if not np.isfinite(t) or t <= 0:
        raise ValueError('Calibration temperature must be finite and positive.')
    return t


def plot_reliability(before, after, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in [None] + CLASS_NAMES:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], '--', color='gray')
        for label, metrics in [('Original softmax', before), ('Temperature scaled', after)]:
            rows = metrics['bins'] if class_name is None else metrics['classwise'][class_name]['bins']
            rows = [b for b in rows if b['count']]
            ax.plot([b['confidence'] for b in rows], [b['accuracy'] for b in rows], 'o-', label=label)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted probability', ylabel='Observed frequency / accuracy',
               title='Top-label reliability' if class_name is None else f'{class_name}: one-vs-rest reliability')
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"reliability_{class_name or 'top_label'}.png", dpi=150)
        plt.close(fig)
