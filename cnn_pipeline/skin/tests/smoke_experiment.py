"""One-epoch integration smoke test on a controlled local real-image subset.

This checks execution, not clinical efficacy. Existing baseline training provenance
is unknown; any legacy comparison is descriptive and may include training images.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.common import BASE_DIR, CLASS_NAMES, file_sha256, save_json, timestamp
from core.data_pipeline import load_metadata
from train.train_ham10000 import train_model
from research.evaluation import run_calibration, run_clinical, run_comparison
from research.robustness import run_robustness, CORRUPTIONS
from skin_inference import predict_skin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir')
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.results_dir) if args.results_dir else BASE_DIR / 'results' / f'smoke_{timestamp()}'
    if root.exists():
        raise FileExistsError(f'Smoke directory already exists: {root}')
    root.mkdir(parents=True)
    baseline = BASE_DIR / 'weights/model_ham10000_best.pth'
    baseline_hash = file_sha256(baseline)
    frame, images = load_metadata(BASE_DIR / 'data')
    selected_lesions = []
    for name in CLASS_NAMES:
        lesions = sorted(frame[frame.dx == name].lesion_id.unique())
        if len(lesions) < 8:
            raise ValueError('Smoke test needs at least eight available lesions per class.')
        selected_lesions.extend(lesions[:8])
    subset = frame[frame.lesion_id.isin(selected_lesions)].reset_index(drop=True)
    data = root / 'data'; (data / 'images').mkdir(parents=True)
    subset.to_csv(data / 'HAM10000_metadata.csv', index=False)
    for image_id in subset.image_id:
        shutil.copyfile(images / f'{image_id}.jpg', data / 'images' / f'{image_id}.jpg')
    checkpoint = root / 'weights/model_ham10000_upgraded.pth'
    train_model(data, checkpoint, epochs=1, batch_size=4, pretrained=False, seed=123,
                validation_size=.25, test_size=.25, results_dir=root / 'training', device='cpu')
    split_path = root / 'training/splits.json'
    shared = {'checkpoint': str(checkpoint), 'splits': str(split_path), 'data_dir': str(data),
              'device': 'cpu', 'batch_size': 4, 'seed': 123, 'n_bins': 10, 'bootstrap': 0,
              'allow_unverified_checkpoint': False}
    artifact = checkpoint.with_name('calibration_temperature.json')
    run_calibration(SimpleNamespace(**shared, command='calibrate', results_dir=str(root / 'calibration'),
                                    output=str(artifact), specificity_targets=[.9, .95, .99], sensitivity_targets=[.9, .95, .99]))
    clinical_shared = dict(shared, bootstrap=30)
    run_clinical(SimpleNamespace(**clinical_shared, command='clinical', results_dir=str(root / 'clinical'),
                                 calibration=str(artifact), thresholds=str(root / 'calibration/metrics/operating_points.json'),
                                 specificity_targets=[.9, .95, .99], sensitivity_targets=[.9, .95, .99]))
    run_comparison(SimpleNamespace(**clinical_shared, command='compare', results_dir=str(root / 'comparison'),
                                   calibration=str(artifact), baseline=str(baseline), baseline_calibration=None,
                                   allow_unverified_baseline=True))
    run_robustness(SimpleNamespace(**shared, command='robustness', results_dir=str(root / 'robustness'),
                                   calibration=str(artifact), corruptions=CORRUPTIONS, severity=1,
                                   attacks=['fgsm', 'pgd'], epsilon=2/255, iterations=2, step_size=None))
    # Independently reload and check inference interface on the trained structured checkpoint.
    output = predict_skin(data / 'images' / f'{subset.iloc[0].image_id}.jpg', checkpoint, 'cpu')
    assert len(output['differential']) == 7
    robustness_report = json.loads((root / 'robustness/metrics/robustness.json').read_text())
    assert robustness_report['weights_unchanged']
    assert baseline_hash == file_sha256(baseline), 'Baseline checkpoint changed!'
    save_json(root / 'reports/smoke_summary.json', {'purpose': 'Integration smoke test, not a research result',
                                                  'images': len(subset), 'lesions': subset.lesion_id.nunique(),
                                                  'epochs': 1, 'pretrained': False, 'device': 'cpu',
                                                  'baseline_sha256_before_and_after': baseline_hash,
                                                  'baseline_split_provenance': 'unknown; descriptive comparison only',
                                                  'checks': ['training', 'checkpoint reload', 'validation calibration', 'clinical evaluation',
                                                             'paired comparison', 'lesion bootstrap', 'all ten corruptions',
                                                             'FGSM', 'PGD', 'inference schema', 'baseline unchanged']})
    print(f'Smoke test complete: {root.resolve()}')


if __name__ == '__main__':
    main()
