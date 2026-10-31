"""Deterministic synthetic dermoscopy corruptions and optional pixel-constrained attacks."""
import argparse
import hashlib
import io
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

if __package__ and '.' in __package__:
    from ..core.common import CLASS_NAMES, MEAN, STD, save_json, model_state_sha256
    from ..core.data_pipeline import HAM10000Dataset
    from ..core.augmentation import get_eval_transform
    from .evaluation import (add_shared_arguments, prepare_evaluation, collect_logits,
                             evaluation_temperature, save_predictions)
    from ..core.calibration import calibration_metrics, probabilities_from_logits
    from .clinical_metrics import clinical_metrics
else:
    from core.common import CLASS_NAMES, MEAN, STD, save_json, model_state_sha256
    from core.data_pipeline import HAM10000Dataset
    from core.augmentation import get_eval_transform
    from research.evaluation import (add_shared_arguments, prepare_evaluation, collect_logits,
                            evaluation_temperature, save_predictions)
    from core.calibration import calibration_metrics, probabilities_from_logits
    from research.clinical_metrics import clinical_metrics

CORRUPTIONS = ['hair', 'illumination', 'brightness', 'contrast', 'noise', 'blur', 'jpeg', 'color', 'vignetting', 'uneven_lighting']


def stable_seed(seed, image_id, corruption):
    return int.from_bytes(hashlib.sha256(f'{seed}:{image_id}:{corruption}'.encode()).digest()[:8], 'little')


def hair_overlay(image, rng, severity):
    """Antialiased curved brown hairs with coherent orientation and soft partial opacity."""
    w, h = image.size
    scale = 2
    overlay = image.copy()
    orientation = rng.uniform(0, np.pi)
    for _ in range(3 + 2 * severity):
        angle = orientation + rng.normal(0, .22)
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        center = rng.uniform([0, 0], [w, h])
        length = rng.uniform(.7, 1.3) * max(w, h)
        start, end = center - direction * length / 2, center + direction * length / 2
        control = center + normal * rng.uniform(-.12, .12) * length
        t = np.linspace(0, 1, 80)[:, None]
        points = (1 - t)**2 * start + 2 * t * (1 - t) * control + t**2 * end
        mask = Image.new('L', (w * scale, h * scale), 0)
        draw = ImageDraw.Draw(mask)
        width = max(1, int(rng.uniform(.6, 1.4) * max(w, h) / 224 * scale))
        draw.line([tuple(p * scale) for p in points], fill=int(rng.uniform(140, 220)), width=width)
        mask = mask.resize((w, h), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(.3))
        shade = int(rng.uniform(20, 60))
        strand = Image.new('RGB', (w, h), (shade, int(shade * .8), int(shade * .65)))
        overlay = Image.composite(strand, overlay, mask)
    return overlay


def corrupt_image(image, corruption, severity=1, seed=42):
    if corruption not in CORRUPTIONS or severity not in range(1, 6):
        raise ValueError('Unknown corruption or severity outside 1..5.')
    image = image.convert('RGB')
    rng = np.random.default_rng(seed)
    if corruption == 'hair':
        return hair_overlay(image, rng, severity)
    if corruption == 'blur':
        return image.filter(ImageFilter.GaussianBlur(.4 * severity))
    if corruption == 'jpeg':
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95 - 15 * severity)
        with Image.open(io.BytesIO(buffer.getvalue())) as compressed:
            return compressed.convert('RGB')
    if corruption in ('brightness', 'contrast'):
        enhancer = ImageEnhance.Brightness if corruption == 'brightness' else ImageEnhance.Contrast
        factor = 1 + rng.choice([-1, 1]) * .12 * severity
        return enhancer(image).enhance(factor)
    rgb = np.asarray(image, dtype=np.float32) / 255
    h, w = rgb.shape[:2]
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    if corruption == 'noise':
        rgb += rng.normal(0, .012 * severity, rgb.shape)
    elif corruption == 'illumination':
        rgb *= 1 - .12 * severity
    elif corruption == 'color':
        shift = rng.choice([-1, 1]) * .025 * severity
        rgb *= np.array([1 + shift, 1, 1 - shift], dtype=np.float32)
    elif corruption == 'vignetting':
        field = 1 - .075 * severity * (x**2 + y**2)
        rgb *= field[..., None]
    else:
        angle = rng.uniform(0, 2 * np.pi)
        field = 1 + .12 * severity * (x * np.cos(angle) + y * np.sin(angle)) / np.sqrt(2)
        rgb *= field[..., None]
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype('uint8'))


class CorruptedDataset(HAM10000Dataset):
    def __init__(self, frame, img_dir, corruption, severity, seed):
        super().__init__(frame, img_dir, transform=None)
        self.corruption, self.severity, self.seed = corruption, severity, seed
        self.clean_transform = get_eval_transform()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        image_id = self.df.iloc[idx].image_id
        seed = stable_seed(self.seed, image_id, self.corruption)
        return self.clean_transform(corrupt_image(image, self.corruption, self.severity, seed)), label


def adversarial_attack(model, normalized_images, labels, attack='fgsm', epsilon=2/255, iterations=10, step_size=None):
    """L-infinity attacks in [0,1] RGB pixel space, with gradients through normalization.

    Only input gradients are requested; model parameters and parameter gradients are untouched.
    """
    if attack not in ('fgsm', 'pgd') or not 0 <= epsilon <= 1 or iterations < 1:
        raise ValueError('Choose fgsm/pgd, epsilon in [0,1] and positive iterations.')
    step = float(step_size) if step_size is not None else 2 * epsilon / iterations
    if not np.isfinite(step) or step < 0:
        raise ValueError('Attack step size must be finite and nonnegative.')
    model.eval()
    mean = normalized_images.new_tensor(MEAN)[None, :, None, None]
    std = normalized_images.new_tensor(STD)[None, :, None, None]
    original = (normalized_images.detach() * std + mean).clamp(0, 1)
    if epsilon == 0:
        return normalized_images.detach().clone()
    pixels = original.clone() if attack == 'fgsm' else (original + torch.empty_like(original).uniform_(-epsilon, epsilon)).clamp(0, 1)
    for _ in range(1 if attack == 'fgsm' else iterations):
        pixels = pixels.detach().requires_grad_(True)
        loss = F.cross_entropy(model((pixels - mean) / std), labels)
        gradient, = torch.autograd.grad(loss, pixels, only_inputs=True)
        pixels = pixels.detach() + (epsilon if attack == 'fgsm' else step) * gradient.sign()
        pixels = torch.maximum(torch.minimum(pixels, original + epsilon), original - epsilon).clamp(0, 1)
    return ((pixels - mean) / std).detach()


def collect_corrupted_logits(model, dataset, device, batch_size):
    model.eval()
    outputs, labels = [], []
    with torch.no_grad():
        for images, y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            outputs.append(model(images.to(device)).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def collect_adversarial_logits(model, frame, img_dir, device, batch_size, attack, epsilon, iterations, step_size):
    outputs, labels = [], []
    loader = DataLoader(HAM10000Dataset(frame, img_dir, get_eval_transform()), batch_size=batch_size, shuffle=False)
    for images, y in loader:
        images, targets = images.to(device), y.to(device)
        perturbed = adversarial_attack(model, images, targets, attack, epsilon, iterations, step_size)
        with torch.no_grad():
            outputs.append(model(perturbed).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(outputs), np.concatenate(labels)


def summarize_robustness(labels, logits, temperature, n_bins):
    p = probabilities_from_logits(logits, temperature)
    clinical, calibration = clinical_metrics(labels, p), calibration_metrics(labels, p, n_bins)
    return {k: clinical[k] for k in ['accuracy', 'macro_f1', 'macro_auc']} | {
        'ece': calibration['ece'], 'per_class_sensitivity': {name: clinical['per_class'][name]['sensitivity'] for name in CLASS_NAMES}}


def run_robustness(args):
    if not 0 <= args.epsilon <= 1 or args.iterations < 1 or args.step_size is not None and args.step_size < 0:
        raise ValueError('Invalid adversarial attack settings.')
    model, splits, img_dir, device, out, verified = prepare_evaluation(args)
    initial_model_hash = model_state_sha256(model)
    temperature = evaluation_temperature(args.checkpoint, args.calibration, args.splits)
    frame = splits['test']
    clean_logits, y = collect_logits(model, frame, img_dir, device, args.batch_size)
    clean = summarize_robustness(y, clean_logits, temperature, args.n_bins)
    save_predictions(out / 'metrics/clean_predictions.npz', frame, clean_logits, y, temperature)
    results = {'clean': {'category': 'clean', **clean, 'degradation': {k: 0. for k in ['accuracy', 'macro_f1', 'macro_auc', 'ece']}}}
    for corruption in args.corruptions:
        dataset = CorruptedDataset(frame, img_dir, corruption, args.severity, args.seed)
        logits, corrupted_y = collect_corrupted_logits(model, dataset, device, args.batch_size)
        if not np.array_equal(y, corrupted_y):
            raise ValueError('Corruption evaluation changed sample ordering.')
        metrics = summarize_robustness(y, logits, temperature, args.n_bins)
        results[corruption] = {'category': 'synthetic_corruption', 'severity': args.severity, **metrics,
                              'degradation': {k: metrics[k] - clean[k] if k == 'ece' else clean[k] - metrics[k]
                                              for k in ['accuracy', 'macro_f1', 'macro_auc', 'ece']}}
        save_predictions(out / f'metrics/{corruption}_predictions.npz', frame, logits, y, temperature)
        with Image.open(Path(img_dir) / f'{frame.iloc[0].image_id}.jpg') as source:
            example = corrupt_image(source, corruption, args.severity, stable_seed(args.seed, frame.iloc[0].image_id, corruption))
        (out / 'examples').mkdir(exist_ok=True)
        example.save(out / f'examples/{corruption}.png')
        logging.info('Corruption %s: accuracy %.4f; ECE %.4f', corruption, metrics['accuracy'], metrics['ece'])
    for attack in args.attacks:
        logits, attacked_y = collect_adversarial_logits(model, frame, img_dir, device, args.batch_size,
                                                       attack, args.epsilon, args.iterations, args.step_size)
        if not np.array_equal(y, attacked_y):
            raise ValueError('Attack evaluation changed sample ordering.')
        metrics = summarize_robustness(y, logits, temperature, args.n_bins)
        results[attack] = {'category': 'adversarial', 'epsilon_pixel_0_1': args.epsilon,
                          'epsilon_pixel_0_255': args.epsilon * 255, 'iterations': 1 if attack == 'fgsm' else args.iterations,
                          'step_size': args.epsilon if attack == 'fgsm' else (args.step_size if args.step_size is not None else 2 * args.epsilon / args.iterations),
                          'norm': 'L-infinity', **metrics,
                          'degradation': {k: metrics[k] - clean[k] if k == 'ece' else clean[k] - metrics[k]
                                          for k in ['accuracy', 'macro_f1', 'macro_auc', 'ece']}}
        save_predictions(out / f'metrics/{attack}_predictions.npz', frame, logits, y, temperature)
        logging.info('Adversarial %s: accuracy %.4f; epsilon %.6f', attack, metrics['accuracy'], args.epsilon)
    weights_unchanged = model_state_sha256(model) == initial_model_hash
    if not weights_unchanged:
        raise RuntimeError('Model parameters or buffers changed during robustness evaluation.')
    save_json(out / 'metrics/robustness.json', {'checkpoint_split_verified': verified, 'samples': len(y),
                                              'weights_unchanged': weights_unchanged, 'model_state_sha256': initial_model_hash,
                                              'temperature': temperature, 'conditions': results,
                                              'degradation_definition': 'clean minus disturbed for performance; disturbed minus clean for ECE. Positive means worse.'})
    import pandas as pd
    rows = []
    for name, result in results.items():
        rows.append({'condition': name, 'category': result['category'], **{k: result[k] for k in ['accuracy', 'macro_f1', 'macro_auc', 'ece']},
                     **{f'{k}_degradation': v for k, v in result['degradation'].items()},
                     **{f'{c}_sensitivity': v for c, v in result['per_class_sensitivity'].items()}})
    (out / 'reports').mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'reports/robustness_comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric in zip(axes.ravel(), ['accuracy', 'macro_f1', 'macro_auc', 'ece']):
        ax.bar([r['condition'] for r in rows], [r[metric] for r in rows],
               color=['#555555' if r['category'] == 'clean' else '#b45a3c' if r['category'] == 'adversarial' else '#447c94' for r in rows])
        ax.tick_params(axis='x', rotation=65); ax.set(title=metric, ylabel=metric)
    (out / 'plots').mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out / 'plots/robustness_comparison.png', dpi=150); plt.close(fig)
    logging.info('Robustness results saved to %s', out)


def main():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser, bootstrap=False)
    parser.set_defaults(command='robustness')
    parser.add_argument('--calibration')
    parser.add_argument('--corruptions', nargs='+', choices=CORRUPTIONS, default=CORRUPTIONS)
    parser.add_argument('--severity', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--attacks', nargs='*', choices=['fgsm', 'pgd'], default=[])
    parser.add_argument('--epsilon', type=float, default=2/255, help='L-infinity radius in unnormalized [0,1] RGB pixels')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--step-size', type=float)
    args = parser.parse_args()
    try:
        run_robustness(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
