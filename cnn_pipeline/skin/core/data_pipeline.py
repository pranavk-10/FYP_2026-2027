"""Shared datasets and persisted lesion-disjoint train/validation/test manifests."""
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .common import CLASS_NAMES, NAME_TO_CLASS_INDEX, save_json, file_sha256


class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.img_dir / f"{row['image_id']}.jpg"
        if not path.is_file():
            raise FileNotFoundError(f'Image not found: {path}')
        with Image.open(path) as source:
            image = source.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, NAME_TO_CLASS_INDEX[row['dx']]


def load_metadata(data_dir):
    path = Path(data_dir) / 'HAM10000_metadata.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Metadata missing: {path}. Run train/download_ham10000.py.')
    df = pd.read_csv(path, dtype={'image_id': str, 'lesion_id': str, 'dx': str})
    required = {'image_id', 'lesion_id', 'dx'}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError('Metadata requires non-null image_id, lesion_id and dx.')
    if df.image_id.duplicated().any() or not set(df.dx).issubset(CLASS_NAMES):
        raise ValueError('Metadata contains duplicate image IDs or unknown diagnostic classes.')
    if (df.groupby('lesion_id').dx.nunique() > 1).any():
        raise ValueError('A lesion has inconsistent diagnostic labels.')
    img_dir = Path(data_dir) / 'images'
    available = {p.stem for p in img_dir.glob('*.jpg')}
    filtered = df[df.image_id.isin(available)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f'No available images in {img_dir}.')
    if len(filtered) != len(df):
        logging.warning('Only %d/%d metadata images are available; subset results are not full-dataset results.', len(filtered), len(df))
    return filtered, img_dir


def dataset_fingerprint(df):
    records = df[['image_id', 'lesion_id', 'dx']].sort_values('image_id').to_dict('records')
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()


def assert_no_leakage(splits):
    for name, frame in splits.items():
        if frame.empty:
            raise ValueError(f'{name} split is empty; more lesions are required.')
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if set(splits[name].lesion_id) & set(splits[other].lesion_id):
                raise ValueError(f'Lesion leakage between {name} and {other}.')


def split_data(df, validation_size=.15, test_size=.15, seed=42):
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError('Validation/test fractions must be positive and sum to less than 1.')
    lesions = df.groupby('lesion_id', sort=True).first().reset_index()
    notes = []

    def partition(frame, fraction, split_seed, label):
        try:
            return train_test_split(frame, test_size=fraction, random_state=split_seed, stratify=frame.dx)
        except ValueError as error:
            note = f'{label}: stratified group split unavailable ({error}); using reproducible unstratified lesion split.'
            notes.append(note)
            logging.warning(note)
            return train_test_split(frame, test_size=fraction, random_state=split_seed)

    development, test = partition(lesions, test_size, seed, 'test')
    train, validation = partition(development, validation_size / (1 - test_size), seed + 1, 'validation')
    splits = {name: df[df.lesion_id.isin(group.lesion_id)].reset_index(drop=True)
              for name, group in [('train', train), ('validation', validation), ('test', test)]}
    assert_no_leakage(splits)
    for name, frame in splits.items():
        missing = sorted(set(CLASS_NAMES) - set(frame.dx))
        if missing:
            note = f'{name} lacks classes {missing}; corresponding metrics may be undefined.'
            notes.append(note)
            logging.warning(note)
    return splits, notes


def save_splits(path, df, splits, seed, validation_size, test_size, notes):
    statistics = {name: {'images': len(frame), 'lesions': frame.lesion_id.nunique(),
                         'images_per_class': frame.dx.value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
                         'lesions_per_class': frame.groupby('dx').lesion_id.nunique().reindex(CLASS_NAMES, fill_value=0).to_dict()}
                  for name, frame in splits.items()}
    save_json(path, {'dataset_fingerprint': dataset_fingerprint(df), 'seed': seed,
                     'validation_size': validation_size, 'test_size': test_size, 'limitations': notes,
                     'statistics': statistics, 'splits': {k: v[['image_id', 'lesion_id', 'dx']].to_dict('records')
                                                        for k, v in splits.items()}})
    return file_sha256(path)


def load_splits(path, data_dir):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    df, img_dir = load_metadata(data_dir)
    if dataset_fingerprint(df) != manifest['dataset_fingerprint']:
        raise ValueError('Available dataset changed since splitting; use the original dataset and manifest.')
    splits = {name: pd.DataFrame(records) for name, records in manifest['splits'].items()}
    if set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('Manifest must contain train, validation and test splits.')
    combined = pd.concat(splits.values(), ignore_index=True)
    if combined.image_id.duplicated().any() or dataset_fingerprint(combined) != dataset_fingerprint(df):
        raise ValueError('Manifest does not partition the available dataset exactly.')
    assert_no_leakage(splits)
    return splits, img_dir, manifest
