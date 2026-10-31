"""Shared contracts, serialization and checkpoint handling for the skin pipeline."""
import hashlib
import json
import math
import os
import random
import platform
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models

BASE_DIR = Path(__file__).resolve().parents[1]
IMG_SIZE = 224
CLASS_INDEX_TO_NAME = {0: 'akiec', 1: 'bcc', 2: 'bkl', 3: 'df', 4: 'mel', 5: 'nv', 6: 'vasc'}
NAME_TO_CLASS_INDEX = {v: k for k, v in CLASS_INDEX_TO_NAME.items()}
CLASS_NAMES = list(CLASS_INDEX_TO_NAME.values())
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESSING = {'size': IMG_SIZE, 'mean': MEAN, 'std': STD, 'resize': 'bilinear', 'color': 'RGB'}
MODEL_VERSION = 'ham10000-resnet18-research-v1'


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def timestamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')


def runtime_environment():
    return {'python': platform.python_version(), 'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            'versions': {name: version(name) for name in ['torch', 'torchvision', 'numpy', 'pandas',
                                                        'scikit-learn', 'scipy', 'matplotlib']}}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def model_state_sha256(model):
    """Hash parameters and buffers to audit evaluation weight integrity."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_resnet18(pretrained=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def protect_baseline(path):
    path = Path(path).resolve()
    if path.name == 'model_ham10000_best.pth' or path == (BASE_DIR / 'weights/model_ham10000_best.pth').resolve():
        raise ValueError('The baseline checkpoint is protected; choose a separate upgraded filename.')


def load_checkpoint(path, device='cpu'):
    """Load legacy raw state dictionaries and structured checkpoints without unpickling objects."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected a state dictionary or structured checkpoint.')
    mapping = checkpoint.get('class_index_to_name')
    if mapping is not None and {int(k): v for k, v in mapping.items()} != CLASS_INDEX_TO_NAME:
        raise ValueError('Checkpoint class ordering differs from the original HAM10000 mapping.')
    if 'preprocessing' in checkpoint and checkpoint['preprocessing'] != PREPROCESSING:
        raise ValueError('Checkpoint preprocessing differs from the original inference contract.')
    state = next((checkpoint[k] for k in ('model_state_dict', 'state_dict', 'model') if k in checkpoint), checkpoint)
    model = build_resnet18().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    metadata = checkpoint if state is not checkpoint else {}
    return model, metadata
