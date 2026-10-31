"""Multiclass metrics, validation-selected operating points and one-vs-rest utility."""
from pathlib import Path

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, roc_auc_score, roc_curve,
                             precision_recall_curve, average_precision_score)

if __package__ and '.' in __package__:
    from ..core.calibration import validate_probabilities
    from ..core.common import CLASS_NAMES, save_json
else:
    from core.calibration import validate_probabilities
    from core.common import CLASS_NAMES, save_json


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float('nan')


def clinical_metrics(labels, probabilities):
    y, p = validate_probabilities(labels, probabilities)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(7))
    per_class, aucs, aps = {}, [], []
    for c, name in enumerate(CLASS_NAMES):
        tp = int(cm[c, c]); fn = int(cm[c].sum() - tp); fp = int(cm[:, c].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        binary = y == c
        auc = float(roc_auc_score(binary, p[:, c])) if binary.any() and (~binary).any() else float('nan')
        ap = float(average_precision_score(binary, p[:, c])) if binary.any() and (~binary).any() else float('nan')
        aucs.append(auc); aps.append(ap)
        per_class[name] = {'sensitivity': tp / (tp + fn) if tp + fn else float('nan'),
                           'specificity': tn / (tn + fp) if tn + fp else float('nan'),
                           'precision': tp / (tp + fp) if tp + fp else 0., 'support': int(binary.sum()),
                           'roc_auc': auc, 'average_precision': ap, 'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn}
    sensitivity = safe_mean([m['sensitivity'] for m in per_class.values()])
    one_hot = np.eye(7)[y]
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1_score(y, pred, labels=np.arange(7), average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=np.arange(7), average='weighted', zero_division=0)),
            'macro_sensitivity': sensitivity, 'balanced_accuracy': sensitivity,
            'macro_specificity': safe_mean([m['specificity'] for m in per_class.values()]),
            'macro_precision': safe_mean([m['precision'] for m in per_class.values()]),
            'macro_auc': safe_mean(aucs), 'micro_auc': float(roc_auc_score(one_hot.ravel(), p.ravel())),
            'macro_average_precision': safe_mean(aps), 'auc_valid_classes': int(np.isfinite(aucs).sum()),
            'per_class': per_class, 'confusion_matrix': cm}


def select_operating_points(labels, probabilities, specificity_targets=(.9, .95, .99), sensitivity_targets=(.9, .95, .99), split='validation'):
    if split != 'validation':
        raise ValueError('Operating thresholds must be selected using validation labels only.')
    if any(not 0 < t <= 1 for t in (*specificity_targets, *sensitivity_targets)):
        raise ValueError('Operating-point targets must lie in (0,1].')
    y, p = validate_probabilities(labels, probabilities)
    output = {'selection_split': 'validation', 'interpretation': 'Independent one-vs-rest decisions, not a multiclass argmax policy.', 'classes': {}}
    for c, name in enumerate(CLASS_NAMES):
        binary = y == c
        points = {}
        if not binary.any() or not (~binary).any():
            output['classes'][name] = {'unavailable': 'Validation positives or negatives missing.'}
            continue
        thresholds = np.r_[np.nextafter(1., 2.), np.unique(p[:, c])[::-1], 0.]
        decisions = p[:, c, None] >= thresholds[None, :]
        sensitivity = decisions[binary].mean(0)
        specificity = (~decisions[~binary]).mean(0)
        for kind, targets in [('specificity', specificity_targets), ('sensitivity', sensitivity_targets)]:
            for target in targets:
                constraint, objective = (specificity, sensitivity) if kind == 'specificity' else (sensitivity, specificity)
                eligible = np.flatnonzero(constraint >= target)
                best = max(eligible, key=lambda idx: (objective[idx], constraint[idx], thresholds[idx]))
                points[f'{kind}_{target:.2f}'] = {'threshold': float(thresholds[best]),
                                                 'validation_sensitivity': float(sensitivity[best]),
                                                 'validation_specificity': float(specificity[best]), 'target': target}
        output['classes'][name] = points
    return output


def evaluate_operating_points(labels, probabilities, operating_points):
    y, p = validate_probabilities(labels, probabilities)
    if operating_points.get('selection_split') != 'validation':
        raise ValueError('Expected validation-selected thresholds.')
    results = {}
    for c, name in enumerate(CLASS_NAMES):
        binary = y == c
        results[name] = {}
        for key, point in operating_points['classes'][name].items():
            if key == 'unavailable':
                results[name][key] = point
                continue
            decision = p[:, c] >= point['threshold']
            results[name][key] = {**point, 'test_sensitivity': float(decision[binary].mean()) if binary.any() else float('nan'),
                                  'test_specificity': float((~decision[~binary]).mean()) if (~binary).any() else float('nan')}
    return results


def decision_curves(labels, probabilities, thresholds=None):
    y, p = validate_probabilities(labels, probabilities)
    thresholds = np.linspace(.01, .8, 80) if thresholds is None else np.asarray(thresholds, dtype=float)
    if (thresholds <= 0).any() or (thresholds >= 1).any():
        raise ValueError('Decision-curve thresholds must be strictly between 0 and 1.')
    output = {}
    for c, name in enumerate(CLASS_NAMES):
        binary = y == c
        rows = []
        for t in thresholds:
            decisions = p[:, c] >= t
            tp = np.mean(decisions & binary); fp = np.mean(decisions & ~binary)
            prevalence = float(binary.mean())
            rows.append({'threshold': float(t), 'model': float(tp - fp * t / (1 - t)),
                         'treat_all': prevalence - (1 - prevalence) * t / (1 - t), 'treat_none': 0.})
        output[name] = rows
    return output


def save_clinical_plots(labels, probabilities, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    y, p = validate_probabilities(labels, probabilities)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))
    curves, macro_tprs = {}, []
    grid = np.linspace(0, 1, 201)
    for c, name in enumerate(CLASS_NAMES):
        binary = y == c
        if not binary.any() or not (~binary).any():
            curves[name] = {'unavailable': 'Positives or negatives missing.'}
            continue
        fpr, tpr, threshold = roc_curve(binary, p[:, c], drop_intermediate=False)
        precision, recall, pr_threshold = precision_recall_curve(binary, p[:, c])
        roc_ax.plot(fpr, tpr, label=f'{name} (AUC {roc_auc_score(binary, p[:, c]):.3f})')
        pr_ax.plot(recall, precision, label=f'{name} (AP {average_precision_score(binary, p[:, c]):.3f})')
        macro_tprs.append(np.interp(grid, fpr, tpr))
        curves[name] = {'fpr': fpr, 'tpr': tpr, 'roc_thresholds': threshold, 'precision': precision, 'recall': recall, 'pr_thresholds': pr_threshold}
    if macro_tprs:
        roc_ax.plot(grid, np.mean(macro_tprs, axis=0), '--', label='Macro ROC (interpolated)')
    micro_fpr, micro_tpr, _ = roc_curve(np.eye(7)[y].ravel(), p.ravel())
    roc_ax.plot(micro_fpr, micro_tpr, ':', label='Micro ROC')
    roc_ax.plot([0, 1], [0, 1], '--', color='gray')
    for fig, ax, title, x, ylabel, filename in [(roc_fig, roc_ax, 'One-vs-rest ROC', 'False positive rate', 'Sensitivity', 'roc.png'),
                                                (pr_fig, pr_ax, 'One-vs-rest precision-recall', 'Recall', 'Precision', 'precision_recall.png')]:
        ax.set(xlabel=x, ylabel=ylabel, title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    save_json(output_dir / 'curve_data.json', curves)
    dca = decision_curves(y, p)
    save_json(output_dir / 'decision_curves.json', {'interpretation': 'One-vs-rest net benefit; exploratory and not evidence of patient outcome improvement.', 'classes': dca})
    pd.DataFrame([{'class': name, **row} for name, rows in dca.items() for row in rows]).to_csv(output_dir / 'decision_curves.csv', index=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, rows) in zip(axes.ravel(), dca.items()):
        for strategy in ['model', 'treat_all', 'treat_none']:
            ax.plot([r['threshold'] for r in rows], [r[strategy] for r in rows], label=strategy)
        ax.set(title=f'{name} vs rest', xlabel='Threshold probability', ylabel='Net benefit')
        ax.legend(fontsize=8)
    axes.ravel()[-1].axis('off')
    fig.tight_layout(); fig.savefig(output_dir / 'decision_curves.png', dpi=150); plt.close(fig)
