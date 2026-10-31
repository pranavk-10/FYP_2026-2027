"""Fast mathematical/contract tests; the separate smoke runner exercises real training."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.common import (CLASS_NAMES, CLASS_INDEX_TO_NAME, BASE_DIR, MEAN, STD,
                    load_checkpoint, save_json, protect_baseline, seed_everything)
from core.augmentation import get_transforms, MildElastic, MildMorphology
from core.calibration import (probabilities_from_logits, calibration_metrics, fit_temperature,
                         reliability_bins, save_temperature, load_temperature)
from research.clinical_metrics import (clinical_metrics, select_operating_points, evaluate_operating_points,
                              decision_curves)
from core.data_pipeline import load_metadata, split_data, save_splits, load_splits, assert_no_leakage, HAM10000Dataset
from research.statistical_tests import mcnemar_test, bootstrap_confidence_intervals
from research.robustness import CORRUPTIONS, corrupt_image, adversarial_attack
from research.evaluation import verify_checkpoint_split
import skin_inference


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        seed_everything(42)

    def test_mapping_and_dataset(self):
        self.assertEqual(CLASS_NAMES, ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'])
        frame, images = load_metadata(BASE_DIR / 'data')
        for c, name in enumerate(CLASS_NAMES):
            row = frame[frame.dx == name].iloc[:1]
            image, label = HAM10000Dataset(row, images, get_transforms('none')[0])[0]
            self.assertEqual(label, c)
            self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_compatibility_entrypoints_share_implementations(self):
        import evaluation
        import robustness
        from research import evaluation as canonical_evaluation
        from research import robustness as canonical_robustness
        from train.train_ham10000 import build_model
        from train.download_ham10000 import get_default_data_dir
        self.assertIs(evaluation.run_calibration, canonical_evaluation.run_calibration)
        self.assertIs(robustness.adversarial_attack, canonical_robustness.adversarial_attack)
        self.assertEqual(BASE_DIR, Path(__file__).resolve().parents[1])
        self.assertEqual(Path(get_default_data_dir()), BASE_DIR / 'data')
        self.assertEqual(build_model(pretrained=False).fc.out_features, 7)

    def test_splits_disjoint_and_reproducible(self):
        frame, _ = load_metadata(BASE_DIR / 'data')
        splits, notes = split_data(frame, .2, .2, 17)
        repeated, _ = split_data(frame, .2, .2, 17)
        assert_no_leakage(splits)
        for key in splits:
            self.assertEqual(splits[key].image_id.tolist(), repeated[key].image_id.tolist())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.json'
            fingerprint = save_splits(path, frame, splits, 17, .2, .2, notes)
            restored, _, _ = load_splits(path, BASE_DIR / 'data')
            self.assertEqual(len(restored['test']), len(splits['test']))
            self.assertTrue(verify_checkpoint_split({'split_sha256': fingerprint, 'test_used_for_training': False}, path))
            with self.assertRaises(ValueError):
                verify_checkpoint_split({'split_sha256': 'wrong', 'test_used_for_training': False}, path)
            manifest = json.loads(path.read_text())
            manifest['splits']['test'][0]['lesion_id'] = splits['train'].iloc[0].lesion_id
            save_json(path, manifest)
            with self.assertRaises(ValueError):
                load_splits(path, BASE_DIR / 'data')

    def test_small_group_split_reports_limitations(self):
        frame = pd.DataFrame([{'image_id': f'{name}_{i}', 'lesion_id': f'{name}_{i}', 'dx': name}
                              for name in CLASS_NAMES for i in range(2)])
        splits, notes = split_data(frame, .2, .2, 42)
        assert_no_leakage(splits)
        self.assertTrue(any('unstratified' in note for note in notes))

    def test_augmentation_and_clean_determinism(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (80, 100, 3), dtype=np.uint8))
        for policy in ['none', 'baseline', 'custom', 'randaugment', 'autoaugment', 'trivialaugment']:
            training, clean = get_transforms(policy, 1., 1.)
            tensor = training(image)
            self.assertEqual(tuple(tensor.shape), (3, 224, 224))
            self.assertTrue(torch.isfinite(tensor).all())
            pixels = tensor * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
            self.assertGreaterEqual(float(pixels.min()), -1e-6)
            self.assertLessEqual(float(pixels.max()), 1 + 1e-6)
            self.assertTrue(torch.equal(clean(image), clean(image)))
            self.assertTrue(torch.equal(clean(image), skin_inference.get_skin_transform()(image)))
        self.assertEqual(MildElastic(1)(image).mode, 'RGB')
        self.assertEqual(MildMorphology(1)(image).mode, 'RGB')

    def test_calibration_known_values_and_boundaries(self):
        y = np.arange(7)
        perfect = np.eye(7)
        metrics = calibration_metrics(y, perfect, 10)
        for key in ['ece', 'mce', 'brier_score', 'nll']:
            self.assertAlmostEqual(metrics[key], 0)
        uniform = np.full((7, 7), 1/7)
        metrics = calibration_metrics(y, uniform)
        self.assertAlmostEqual(metrics['brier_score'], 6/7)
        self.assertAlmostEqual(metrics['nll'], np.log(7))
        bins = reliability_bins(np.array([0, .5, 1]), np.array([0, 1, 1]), 2)
        self.assertEqual([b['count'] for b in bins], [2, 1])

    def test_temperature_validation_only(self):
        rng = np.random.default_rng(12)
        y = np.tile(np.arange(7), 10)
        logits = rng.normal(0, 5, (len(y), 7))
        temperature, _ = fit_temperature(logits, y)
        before = probabilities_from_logits(logits)
        after = probabilities_from_logits(logits, temperature)
        self.assertTrue(np.allclose(after.sum(1), 1))
        self.assertTrue(np.array_equal(before.argmax(1), after.argmax(1)))
        self.assertLessEqual(calibration_metrics(y, after)['nll'], calibration_metrics(y, before)['nll'])
        with self.assertRaises(ValueError):
            fit_temperature(logits, y, split='test')

    def test_clinical_metrics_and_validation_thresholds(self):
        y = np.repeat(np.arange(7), 2)
        predictions = y.copy(); predictions[1] = 1; predictions[3] = 0
        p = np.eye(7)[predictions] * .94 + .06 / 7
        metrics = clinical_metrics(y, p)
        self.assertAlmostEqual(metrics['accuracy'], 12/14)
        self.assertAlmostEqual(metrics['per_class']['akiec']['sensitivity'], .5)
        self.assertAlmostEqual(metrics['per_class']['akiec']['specificity'], 11/12)
        self.assertAlmostEqual(metrics['balanced_accuracy'], 6/7)
        perfect = np.eye(7)[y]
        self.assertAlmostEqual(clinical_metrics(y, perfect)['macro_auc'], 1.)
        points = select_operating_points(y, perfect)
        evaluated = evaluate_operating_points(y, perfect, points)
        self.assertEqual(evaluated['mel']['specificity_0.95']['test_sensitivity'], 1.)
        with self.assertRaises(ValueError):
            select_operating_points(y, p, split='test')
        curves = decision_curves(y, perfect, [.5])
        self.assertAlmostEqual(curves['mel'][0]['model'], 1/7)
        self.assertAlmostEqual(curves['mel'][0]['treat_all'], -5/7)

    def test_mcnemar_and_paired_cluster_bootstrap(self):
        y = np.repeat(np.arange(7), 4)
        baseline = y.copy(); baseline[:6] = (baseline[:6] + 1) % 7
        test = mcnemar_test(y, baseline, y)
        self.assertEqual(test['contingency_table'], [[22, 0], [6, 0]])
        self.assertAlmostEqual(test['p_value'], .03125)
        p = .9 * np.eye(7)[y] + .1 / 7
        lesions = np.repeat(np.arange(14), 2)
        intervals = bootstrap_confidence_intervals(y, p, lesions, 30, 14, paired_probabilities=p)
        repeated = bootstrap_confidence_intervals(y, p, lesions, 30, 14, paired_probabilities=p)
        self.assertEqual(intervals['baseline']['accuracy'], repeated['baseline']['accuracy'])
        self.assertEqual(intervals['paired_delta_upgraded_minus_baseline']['accuracy']['lower'], 0.)
        self.assertEqual(intervals['paired_delta_upgraded_minus_baseline']['accuracy']['upper'], 0.)
        self.assertEqual(intervals['baseline']['accuracy']['estimate'], 1.)

    def test_corruptions_are_reproducible(self):
        image = Image.fromarray(np.random.default_rng(12).integers(30, 220, (64, 64, 3), dtype=np.uint8))
        for name in CORRUPTIONS:
            first, second = corrupt_image(image, name, 2, 23), corrupt_image(image, name, 2, 23)
            self.assertEqual(first.mode, 'RGB')
            self.assertEqual(first.size, image.size)
            self.assertTrue(np.array_equal(first, second))
            self.assertFalse(np.array_equal(first, image))

    def test_attacks_respect_pixel_constraints_and_weights(self):
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten(), torch.nn.Linear(3, 7))
        model.eval()
        original_weights = {k: v.clone() for k, v in model.state_dict().items()}
        pixels = torch.rand(2, 3, 16, 16)
        mean, std = torch.tensor(MEAN)[None, :, None, None], torch.tensor(STD)[None, :, None, None]
        normalized = (pixels - mean) / std
        for attack in ['fgsm', 'pgd']:
            perturbed = adversarial_attack(model, normalized, torch.tensor([0, 1]), attack, .02, 3)
            attacked_pixels = perturbed * std + mean
            self.assertLessEqual(float((attacked_pixels - pixels).abs().max()), .020001)
            self.assertGreater(float((attacked_pixels - pixels).abs().max()), 0)
            self.assertGreaterEqual(float(attacked_pixels.min()), -1e-6)
            self.assertLessEqual(float(attacked_pixels.max()), 1 + 1e-6)
            self.assertTrue(all(torch.equal(v, original_weights[k]) for k, v in model.state_dict().items()))
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_legacy_checkpoint_and_inference_schema(self):
        model, metadata = load_checkpoint(BASE_DIR / 'weights/model_ham10000_best.pth')
        self.assertEqual(metadata, {})
        self.assertEqual(model.fc.out_features, 7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test_model.pth'
            torch.save({'state_dict': model.state_dict(), 'class_index_to_name': CLASS_INDEX_TO_NAME}, path)
            split_path = Path(directory) / 'splits.json'
            save_json(split_path, {'test': 'artifact identity fixture'})
            self.assertEqual(load_temperature(path), 1.)
            save_temperature(path.with_suffix('.calibration.json'), 2., path, split_path)
            image = Image.new('RGB', (70, 80), color=(140, 105, 95))
            output = skin_inference.predict_skin(image, path, 'cpu')
            self.assertEqual(set(output), {'modality', 'input_type', 'prediction', 'differential', 'evidence', 'explainability'})
            self.assertEqual(set(output['prediction']), {'primary_diagnosis', 'raw_class', 'probability', 'calibrated_probability', 'findings'})
            self.assertEqual(output['evidence'], {'positive_findings': [], 'negative_findings': []})
            self.assertEqual(len(output['differential']), 7)
            self.assertEqual(output['prediction']['findings'], output['differential'])
            for key in ['probability', 'calibrated_probability']:
                self.assertAlmostEqual(sum(item[key] for item in output['differential']), 1., places=6)
            raw = [item['probability'] for item in output['differential']]
            calibrated = [item['calibrated_probability'] for item in output['differential']]
            self.assertFalse(np.allclose(raw, calibrated))
            self.assertEqual(calibrated, sorted(calibrated, reverse=True))
            artifact = json.loads(path.with_suffix('.calibration.json').read_text())
            artifact['checkpoint_sha256'] = 'wrong'
            save_json(path.with_suffix('.calibration.json'), artifact)
            with self.assertRaises(ValueError):
                load_temperature(path)
        with self.assertRaises(ValueError):
            protect_baseline(BASE_DIR / 'weights/model_ham10000_best.pth')


if __name__ == '__main__':
    unittest.main(verbosity=2)
