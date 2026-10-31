"""Shared dermoscopy transforms: conservative training augmentation and fixed clean evaluation."""
import random

import numpy as np
from PIL import Image
from scipy import ndimage
from torchvision import transforms
from torchvision.transforms import functional as TF, InterpolationMode

from .common import IMG_SIZE, MEAN, STD


def get_eval_transform():
    return transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


class MildMorphology:
    """Blend 15% of a 3x3 luminance operation, preserving chroma and fine structure."""
    def __init__(self, probability=0.12, blend=0.15):
        if not 0 <= probability <= 1 or not 0 <= blend <= 0.15:
            raise ValueError('Morphology requires probability in [0,1] and blend in [0,.15].')
        self.probability, self.blend = probability, blend

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        rgb = np.asarray(image, dtype=np.float32)
        luminance = rgb @ np.array([.299, .587, .114], dtype=np.float32)
        operation = random.choice([ndimage.grey_dilation, ndimage.grey_erosion,
                                   ndimage.grey_opening, ndimage.grey_closing])
        modified = operation(luminance, size=(3, 3), mode='reflect')
        output = rgb + self.blend * (modified - luminance)[..., None]
        return Image.fromarray(np.clip(output, 0, 255).astype('uint8'))


class MildElastic:
    """Smooth displacement capped at 1.5 pixels at 224px, bilinear with reflected boundaries."""
    def __init__(self, probability=0.15, max_pixels=1.5):
        if not 0 <= probability <= 1 or not 0 <= max_pixels <= 2:
            raise ValueError('Elastic probability must be in [0,1]; displacement must be <=2px.')
        self.probability, self.max_pixels = probability, max_pixels

    def __call__(self, image):
        if random.random() >= self.probability:
            return image
        arr = np.asarray(image)
        h, w = arr.shape[:2]
        y, x = np.mgrid[:h, :w].astype(float)
        fields = []
        for _ in range(2):
            field = ndimage.gaussian_filter(np.random.uniform(-1, 1, (h, w)), sigma=12, mode='reflect')
            fields.append(field * self.max_pixels / max(np.abs(field).max(), 1e-8))
        coords = [y + fields[0], x + fields[1]]
        out = np.stack([ndimage.map_coordinates(arr[..., c], coords, order=1, mode='reflect')
                        for c in range(3)], axis=-1)
        return Image.fromarray(out.astype('uint8'))


def lighting(image, magnitude):
    rgb = np.asarray(image, dtype=np.float32) / 255
    h, w = rgb.shape[:2]
    y, x = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    angle = random.uniform(0, 2 * np.pi)
    field = 1 + magnitude * (x * np.cos(angle) + y * np.sin(angle)) / np.sqrt(2)
    rgb *= field[..., None] * random.uniform(1 - magnitude, 1 + magnitude)
    return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype('uint8'))


def color_temperature(image, magnitude):
    rgb = np.asarray(image, dtype=np.float32)
    shift = random.uniform(-magnitude, magnitude)
    rgb *= np.array([1 + shift, 1, 1 - shift], dtype=np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype('uint8'))


class ControlledPolicy:
    """Rand/Auto/Trivial policy patterns restricted to safe photometric operations.

    These are dermoscopy adaptations, not torchvision's unrestricted ImageNet policies;
    no solarization, posterization, cutout or aggressive warping is used.
    """
    operations = ('brightness', 'contrast', 'saturation', 'hue', 'gamma', 'lighting', 'temperature')

    def __init__(self, policy='custom'):
        if policy not in ('custom', 'randaugment', 'autoaugment', 'trivialaugment'):
            raise ValueError(f'Unknown controlled policy: {policy}')
        self.policy = policy

    def apply(self, image, operation, level):
        if operation in ('brightness', 'contrast', 'saturation'):
            return getattr(TF, 'adjust_' + operation)(image, 1 + random.uniform(-.12, .12) * level)
        if operation == 'hue':
            return TF.adjust_hue(image, random.uniform(-.015, .015) * level)
        if operation == 'gamma':
            return TF.adjust_gamma(image, random.uniform(1 - .1 * level, 1 + .1 * level))
        if operation == 'lighting':
            return lighting(image, .08 * level)
        return color_temperature(image, .035 * level)

    def __call__(self, image):
        if self.policy == 'custom':
            operations = [op for op in self.operations if random.random() < .35]
            level = 1.0
        elif self.policy == 'randaugment':
            operations, level = random.sample(self.operations, 2), .7
        elif self.policy == 'autoaugment':
            operations = random.choice([('brightness', 'gamma'), ('contrast', 'temperature'),
                                        ('saturation', 'lighting'), ('hue', 'brightness')])
            operations, level = [op for op in operations if random.random() < .6], .7
        else:
            operations, level = [random.choice(self.operations)], random.uniform(.1, 1)
        for operation in operations:
            image = self.apply(image, operation, level)
        return image


def get_transforms(policy='custom', morphology_probability=.12, elastic_probability=.15):
    if policy not in ('none', 'baseline', 'custom', 'randaugment', 'autoaugment', 'trivialaugment'):
        raise ValueError(f'Unknown augmentation policy: {policy}')
    clean = get_eval_transform()
    if policy == 'none':
        return clean, clean
    stages = [transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.RandomHorizontalFlip(.5),
              transforms.RandomVerticalFlip(.5)]
    if policy == 'baseline':
        stages += [transforms.RandomRotation(20), transforms.ColorJitter(.1, .1, .1)]
    else:
        stages += [transforms.RandomRotation(15, interpolation=InterpolationMode.BILINEAR,
                                            fill=(124, 116, 104)), ControlledPolicy(policy),
                   MildMorphology(morphology_probability), MildElastic(elastic_probability)]
    stages += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(stages), clean
