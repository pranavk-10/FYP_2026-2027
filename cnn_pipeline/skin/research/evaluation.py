"""CLI for calibration, final clinical evaluation and paired model comparison."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ and '.' in __package__:
    from ..core.common import BASE_DIR, CLASS_NAMES, seed_everything, timestamp, save_json, file_sha256, load_checkpoint, runtime_environment
    from ..core.data_pipeline import HAM10000Dataset, load_splits
    from ..core.augmentation import get_eval_transform
    from ..core.calibration import (fit_temperature, save_temperature, load_temperature, probabilities_from_logits,
                              calibration_metrics, plot_reliability)
    from .clinical_metrics import clinical_metrics, select_operating_points, evaluate_operating_points, save_clinical_plots
    from .statistical_tests import mcnemar_test, bootstrap_confidence_intervals, metric_vector
else:
    from core.common import BASE_DIR, CLASS_NAMES, seed_everything, timestamp, save_json, file_sha256, load_checkpoint, runtime_environment
    from core.data_pipeline import HAM10000Dataset, load_splits
    from core.augmentation import get_eval_transform
    from core.calibration import (fit_temperature, save_temperature, load_temperature, probabilities_from_logits,
                             calibration_metrics, plot_reliability)
    from research.clinical_metrics import clinical_metrics, select_operating_points, evaluate_operating_points, save_clinical_plots
    from research.statistical_tests import mcnemar_test, bootstrap_confidence_intervals, metric_vector


def verify_checkpoint_split(metadata, split_path, allow_unverified=False):
    if metadata.get('split_sha256'):
        if metadata['split_sha256'] != file_sha256(split_path):
            raise ValueError('Checkpoint was trained on a different split manifest.')
        if metadata.get('test_used_for_training') is not False:
            raise ValueError('Checkpoint does not attest to a held-out test partition.')
        return True
    if not allow_unverified:
        raise ValueError('Legacy checkpoint has no split provenance. Use --allow-unverified-checkpoint only for descriptive/smoke evaluation.')
    logging.warning('Checkpoint split provenance is unknown; results cannot establish unbiased held-out performance.')
    return False


def collect_logits(model, frame, img_dir, device, batch_size=32, transform=None):
    if batch_size < 1:
        raise ValueError('Batch size must be positive.')
    loader = DataLoader(HAM10000Dataset(frame, img_dir, transform or get_eval_transform()),
                        batch_size=batch_size, shuffle=False, num_workers=0)
    logits, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, target in loader:
            logits.append(model(images.to(device)).cpu().numpy())
            labels.append(target.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def prepare_evaluation(args):
    if args.n_bins < 1 or args.bootstrap < 0 or args.batch_size < 1:
        raise ValueError('n-bins/batch-size must be positive; bootstrap must be nonnegative.')
    seed_everything(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    splits, img_dir, manifest = load_splits(args.splits, args.data_dir)
    model, metadata = load_checkpoint(args.checkpoint, device)
    verified = verify_checkpoint_split(metadata, args.splits, args.allow_unverified_checkpoint)
    output_dir = Path(args.results_dir) if args.results_dir else BASE_DIR / 'results' / f'{args.command}_{timestamp()}'
    if (output_dir / 'evaluation_config.json').exists():
        raise FileExistsError('Results directory already contains an evaluation; choose a new --results-dir.')
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), 'timestamp': timestamp(), 'checkpoint_sha256': file_sha256(args.checkpoint),
              'split_sha256': file_sha256(args.splits), 'class_names': CLASS_NAMES,
              'checkpoint_split_verified': verified, 'split_statistics': manifest['statistics'],
              'split_limitations': manifest['limitations'], 'device': str(device), 'environment': runtime_environment()}
    save_json(output_dir / 'evaluation_config.json', config)
    return model, splits, img_dir, device, output_dir, verified


def evaluation_temperature(checkpoint, artifact, splits):
    path = Path(artifact) if artifact else (Path(checkpoint).with_name('calibration_temperature.json')
           if Path(checkpoint).name == 'model_ham10000_upgraded.pth' else Path(checkpoint).with_suffix('.calibration.json'))
    if artifact or path.exists():
        return load_temperature(checkpoint, path, split_path=splits)
    logging.info('No calibration artifact: temperature=1.0, probabilities are uncalibrated softmax.')
    return 1.


def save_predictions(path, frame, logits, labels, temperature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, logits=logits, labels=labels, probabilities=probabilities_from_logits(logits),
                        calibrated_probabilities=probabilities_from_logits(logits, temperature),
                        image_ids=frame.image_id.to_numpy(dtype=str), lesion_ids=frame.lesion_id.to_numpy(dtype=str),
                        class_names=np.array(CLASS_NAMES), temperature=temperature)


def run_calibration(args):
    model, splits, img_dir, device, out, verified = prepare_evaluation(args)
    logits, y = collect_logits(model, splits['validation'], img_dir, device, args.batch_size)
    temperature, optimization = fit_temperature(logits, y)
    before = calibration_metrics(y, probabilities_from_logits(logits), args.n_bins)
    after = calibration_metrics(y, probabilities_from_logits(logits, temperature), args.n_bins)
    artifact = Path(args.output) if args.output else (Path(args.checkpoint).with_name('calibration_temperature.json')
               if Path(args.checkpoint).name == 'model_ham10000_upgraded.pth' else Path(args.checkpoint).with_suffix('.calibration.json'))
    if artifact.exists():
        raise FileExistsError(f'Calibration artifact exists: {artifact}; select a new --output.')
    save_temperature(artifact, temperature, args.checkpoint, args.splits,
                     {'validation_samples': len(y), 'optimization': optimization, 'checkpoint_split_verified': verified,
                      'timestamp': timestamp(), 'metrics_before': before, 'metrics_after': after})
    thresholds = select_operating_points(y, probabilities_from_logits(logits, temperature),
                                         args.specificity_targets, args.sensitivity_targets)
    thresholds.update({'checkpoint_sha256': file_sha256(args.checkpoint), 'split_sha256': file_sha256(args.splits),
                       'temperature': temperature, 'class_names': CLASS_NAMES})
    save_json(out / 'metrics/operating_points.json', thresholds)
    save_json(out / 'metrics/calibration.json', {'temperature': temperature, 'before': before, 'after': after})
    save_predictions(out / 'metrics/validation_predictions.npz', splits['validation'], logits, y, temperature)
    plot_reliability(before, after, out / 'plots')
    logging.info('Validation-only calibration: T=%.6f; ECE %.6f -> %.6f; NLL %.6f -> %.6f; artifact=%s',
                 temperature, before['ece'], after['ece'], before['nll'], after['nll'], artifact)


def run_clinical(args):
    model, splits, img_dir, device, out, verified = prepare_evaluation(args)
    temperature = evaluation_temperature(args.checkpoint, args.calibration, args.splits)
    if args.thresholds:
        points = json.loads(Path(args.thresholds).read_text(encoding='utf-8'))
        if points.get('checkpoint_sha256') != file_sha256(args.checkpoint) or points.get('split_sha256') != file_sha256(args.splits) or points.get('class_names') != CLASS_NAMES or points.get('temperature') != temperature:
            raise ValueError('Threshold artifact does not match the model, split, temperature and classes.')
    else:
        validation_logits, validation_y = collect_logits(model, splits['validation'], img_dir, device, args.batch_size)
        points = select_operating_points(validation_y, probabilities_from_logits(validation_logits, temperature),
                                         args.specificity_targets, args.sensitivity_targets)
        points.update({'checkpoint_sha256': file_sha256(args.checkpoint), 'split_sha256': file_sha256(args.splits),
                       'temperature': temperature, 'class_names': CLASS_NAMES})
    save_json(out / 'metrics/selected_operating_points.json', points)
    logits, y = collect_logits(model, splits['test'], img_dir, device, args.batch_size)
    raw = probabilities_from_logits(logits); calibrated = probabilities_from_logits(logits, temperature)
    clinical = clinical_metrics(y, calibrated)
    before, after = calibration_metrics(y, raw, args.n_bins), calibration_metrics(y, calibrated, args.n_bins)
    report = {'evaluation_split': 'test', 'checkpoint_split_verified': verified, 'samples': len(y),
              'temperature': temperature, 'clinical': clinical, 'calibration_before': before, 'calibration_after': after,
              'operating_points': evaluate_operating_points(y, calibrated, points),
              'interpretation': 'OvR operating points were selected on validation. Test target specificity/sensitivity is not guaranteed. DCA is exploratory.'}
    save_json(out / 'metrics/clinical_metrics.json', report)
    save_predictions(out / 'metrics/test_predictions.npz', splits['test'], logits, y, temperature)
    save_clinical_plots(y, calibrated, out / 'plots')
    plot_reliability(before, after, out / 'plots')
    if args.bootstrap:
        interval = bootstrap_confidence_intervals(y, calibrated, splits['test'].lesion_id, args.bootstrap, args.seed, args.n_bins)
        interval['model'] = interval.pop('baseline')
        save_json(out / 'metrics/bootstrap_confidence_intervals.json', interval)
    import pandas as pd
    pd.DataFrame(clinical['per_class']).T.to_csv(out / 'metrics/per_class_metrics.csv')
    save_json(out / 'reports/clinical_summary.json', {k: report[k] for k in ['samples', 'temperature', 'checkpoint_split_verified', 'interpretation']})
    logging.info('Test: accuracy=%.4f macro-F1=%.4f macro-AUC=%.4f ECE=%.4f; saved to %s',
                 clinical['accuracy'], clinical['macro_f1'], clinical['macro_auc'], after['ece'], out)


def run_comparison(args):
    model, splits, img_dir, device, out, upgraded_verified = prepare_evaluation(args)
    baseline, baseline_metadata = load_checkpoint(args.baseline, device)
    baseline_verified = verify_checkpoint_split(baseline_metadata, args.splits, args.allow_unverified_baseline)
    baseline_t = evaluation_temperature(args.baseline, args.baseline_calibration, args.splits)
    upgraded_t = evaluation_temperature(args.checkpoint, args.calibration, args.splits)
    baseline_logits, y = collect_logits(baseline, splits['test'], img_dir, device, args.batch_size)
    upgraded_logits, upgraded_y = collect_logits(model, splits['test'], img_dir, device, args.batch_size)
    if not np.array_equal(y, upgraded_y):
        raise ValueError('Paired predictions have different sample labels/order.')
    bp, up = probabilities_from_logits(baseline_logits, baseline_t), probabilities_from_logits(upgraded_logits, upgraded_t)
    b, u = metric_vector(y, bp, args.n_bins), metric_vector(y, up, args.n_bins)
    test = mcnemar_test(y, bp.argmax(1), up.argmax(1))
    eligible = baseline_verified and upgraded_verified
    rows = []
    for key in ['accuracy', 'macro_f1', 'macro_auc', 'macro_sensitivity', 'macro_specificity', 'ece', 'brier_score']:
        rows.append({'metric': key, 'baseline': b[key], 'upgraded': u[key],
                     'improvement': (b[key] - u[key]) if key in ('ece', 'brier_score') else (u[key] - b[key]),
                     'direction': 'lower is better' if key in ('ece', 'brier_score') else 'higher is better'})
    interval = bootstrap_confidence_intervals(y, bp, splits['test'].lesion_id, args.bootstrap, args.seed, args.n_bins, up) if args.bootstrap else None
    accuracy_interval = interval['paired_delta_upgraded_minus_baseline']['accuracy'] if interval else None
    statistically_supported = bool(eligible and test['significant'] and test['upgraded_accuracy_higher'] and accuracy_interval
                                   and accuracy_interval['lower'] is not None and accuracy_interval['lower'] > 0)
    report = {'samples': len(y), 'same_test_samples': True, 'baseline_checkpoint_sha256': file_sha256(args.baseline),
              'upgraded_checkpoint_sha256': file_sha256(args.checkpoint), 'split_sha256': file_sha256(args.splits),
              'baseline_temperature': baseline_t, 'upgraded_temperature': upgraded_t, 'comparison': rows,
              'mcnemar': test, 'bootstrap': interval, 'held_out_provenance_verified': eligible,
              'statistically_supported_accuracy_improvement': statistically_supported,
              'limitations': ['No unbiased improvement claim is valid for checkpoints with unknown split provenance.',
                              'McNemar is image-paired; lesion-cluster bootstrap accounts for correlated images.',
                              'Multiple metric comparisons are exploratory; confidence intervals are not multiplicity-adjusted.']}
    save_json(out / 'reports/model_comparison.json', report)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / 'reports/model_comparison.csv', index=False)
    lines = ['# Baseline versus upgraded', '', f'Same test images: {len(y)}. Held-out provenance verified: {eligible}.',
             '', '| Metric | Baseline | Upgraded | Improvement |', '|---|---:|---:|---:|']
    lines += [f"| {r['metric']} | {r['baseline']:.6f} | {r['upgraded']:.6f} | {r['improvement']:+.6f} |" for r in rows]
    lines += ['', 'Improvement is direction-adjusted; higher is better. Unknown baseline provenance prevents unbiased claims.',
              f"McNemar exact p={test['p_value']:.6g}; supported accuracy improvement={statistically_supported}."]
    (out / 'reports/model_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save_predictions(out / 'metrics/baseline_test_predictions.npz', splits['test'], baseline_logits, y, baseline_t)
    save_predictions(out / 'metrics/upgraded_test_predictions.npz', splits['test'], upgraded_logits, y, upgraded_t)
    logging.info('Comparison saved to %s; supported accuracy improvement=%s', out, statistically_supported)


def add_shared_arguments(parser, bootstrap=True):
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--splits', required=True)
    parser.add_argument('--data-dir', default=str(BASE_DIR / 'data'))
    parser.add_argument('--results-dir')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-bins', type=int, default=15)
    if bootstrap:
        parser.add_argument('--bootstrap', type=int, default=0, help='Lesion bootstrap repetitions, e.g. 1000; disabled by default')
    else:
        parser.set_defaults(bootstrap=0)
    parser.add_argument('--allow-unverified-checkpoint', action='store_true')


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['calibrate', 'clinical', 'compare']:
        sub = commands.add_parser(command)
        add_shared_arguments(sub, bootstrap=command != 'calibrate')
        if command == 'calibrate':
            sub.add_argument('--output', help='Calibration artifact; defaults to checkpoint sidecar')
        else:
            sub.add_argument('--calibration')
        if command in ('calibrate', 'clinical'):
            sub.add_argument('--specificity-targets', type=float, nargs='+', default=[.9, .95, .99])
            sub.add_argument('--sensitivity-targets', type=float, nargs='+', default=[.9, .95, .99])
        if command == 'clinical':
            sub.add_argument('--thresholds', help='Validation-selected threshold artifact; otherwise fit on validation now')
        if command == 'compare':
            sub.add_argument('--baseline', required=True)
            sub.add_argument('--baseline-calibration')
            sub.add_argument('--allow-unverified-baseline', action='store_true')
    args = parser.parse_args()
    try:
        {'calibrate': run_calibration, 'clinical': run_clinical, 'compare': run_comparison}[args.command](args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
