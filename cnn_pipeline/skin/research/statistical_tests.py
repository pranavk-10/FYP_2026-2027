"""Paired correctness tests and reproducible lesion-cluster bootstrap intervals."""
import numpy as np
from scipy.stats import binomtest, chi2

if __package__ and '.' in __package__:
    from ..core.common import CLASS_NAMES
    from .clinical_metrics import clinical_metrics
    from ..core.calibration import calibration_metrics, validate_probabilities
else:
    from core.common import CLASS_NAMES
    from research.clinical_metrics import clinical_metrics
    from core.calibration import calibration_metrics, validate_probabilities


def mcnemar_test(labels, baseline_predictions, upgraded_predictions, alpha=.05):
    y, baseline, upgraded = map(np.asarray, (labels, baseline_predictions, upgraded_predictions))
    if y.ndim != 1 or y.shape != baseline.shape or y.shape != upgraded.shape or not y.size:
        raise ValueError('McNemar requires paired predictions on the same non-empty samples.')
    a, b = baseline == y, upgraded == y
    table = [[int((a & b).sum()), int((a & ~b).sum())],
             [int((~a & b).sum()), int((~a & ~b).sum())]]
    discordant = table[0][1] + table[1][0]
    statistic = max(0, abs(table[0][1] - table[1][0]) - 1)**2 / discordant if discordant else 0.
    p_value = float(binomtest(table[0][1], discordant, .5).pvalue) if discordant else 1.
    return {'contingency_table': table, 'table_order': '[[both correct, baseline only correct], [upgraded only correct, both wrong]]',
            'statistic': statistic, 'statistic_method': 'Continuity-corrected chi-square',
            'p_value': p_value, 'p_value_method': 'Exact two-sided binomial McNemar',
            'asymptotic_p_value': float(chi2.sf(statistic, 1)), 'alpha': alpha,
            'significant': p_value < alpha, 'upgraded_accuracy_higher': table[1][0] > table[0][1],
            'limitation': 'Image-paired test assumes independent pairs; multiple images per lesion can violate this. See cluster bootstrap.'}


def metric_vector(labels, probabilities, n_bins=15):
    clinical = clinical_metrics(labels, probabilities)
    calibration = calibration_metrics(labels, probabilities, n_bins)
    vector = {k: clinical[k] for k in ['accuracy', 'macro_f1', 'macro_auc', 'macro_sensitivity', 'macro_specificity']}
    vector.update({k: calibration[k] for k in ['ece', 'brier_score']})
    for name in CLASS_NAMES:
        for metric in ['sensitivity', 'specificity']:
            vector[f'{name}_{metric}'] = clinical['per_class'][name][metric]
    return vector


def bootstrap_confidence_intervals(labels, probabilities, lesion_ids, n_bootstrap=1000, seed=42, n_bins=15,
                                   paired_probabilities=None):
    y, p = validate_probabilities(labels, probabilities)
    groups = np.asarray(lesion_ids)
    if groups.shape != y.shape or n_bootstrap < 1:
        raise ValueError('Expected lesion IDs aligned with labels and positive bootstrap repetitions.')
    if paired_probabilities is not None:
        _, paired_probabilities = validate_probabilities(y, paired_probabilities)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    point = metric_vector(y, p, n_bins)
    paired_point = metric_vector(y, paired_probabilities, n_bins) if paired_probabilities is not None else None
    samples = {k: [] for k in point}; paired = {k: [] for k in point}; deltas = {k: [] for k in point}
    original_classes = set(y.tolist())
    for _ in range(n_bootstrap):
        idx = np.concatenate([indices[g] for g in rng.choice(unique, size=len(unique), replace=True)])
        first = metric_vector(y[idx], p[idx], n_bins)
        second = metric_vector(y[idx], paired_probabilities[idx], n_bins) if paired_probabilities is not None else None
        # Do not let a changing class subset redefine a macro metric between replicates/models.
        if set(y[idx].tolist()) != original_classes or len(np.unique(y[idx])) < 2:
            for metric in ['macro_auc', 'macro_sensitivity', 'macro_specificity']:
                first[metric] = float('nan')
                if second is not None:
                    second[metric] = float('nan')
        for key, value in first.items():
            samples[key].append(value)
            if second is not None:
                paired[key].append(second[key]); deltas[key].append(second[key] - value)

    def intervals(values, estimates):
        output = {}
        for key, draws in values.items():
            finite = np.asarray(draws)[np.isfinite(draws)]
            output[key] = {'estimate': estimates[key], 'lower': float(np.percentile(finite, 2.5)) if len(finite) else None,
                           'upper': float(np.percentile(finite, 97.5)) if len(finite) else None,
                           'valid_resamples': len(finite), 'invalid_resamples': n_bootstrap - len(finite)}
        return output

    result = {'unit': 'lesion_id cluster (all images retained; image-weighted metrics)', 'seed': seed,
              'repetitions': n_bootstrap, 'confidence': .95, 'method': 'Percentile cluster bootstrap',
              'missing_class_handling': 'Undefined class metrics excluded; macro AUC/sensitivity/specificity draws missing original classes excluded.',
              'baseline': intervals(samples, point)}
    if paired_point is not None:
        result['upgraded'] = intervals(paired, paired_point)
        result['paired_delta_upgraded_minus_baseline'] = intervals(deltas, {k: paired_point[k] - point[k] for k in point})
    return result
