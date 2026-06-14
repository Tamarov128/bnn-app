"""
core/data/transforms.py
───────────────────────
Shared torchvision transform pipelines.

All datasets — in-distribution and OOD alike — arrive at the model as
single-channel 28x28 tensors normalised with MNIST statistics:
    mean = 0.1307,  std = 0.3081

get_mnist_transform()
    Standard pipeline for MNIST, FashionMNIST, and KMNIST.

get_omniglot_transform()
    Pipeline for Omniglot.  Handles:
      - RGB → greyscale conversion  (background images are RGB PNGs)
      - 105x105 → 28x28 resize
      - Pixel inversion (Omniglot is white-on-black; MNIST is black-on-white)
    Normalisation parameters are identical to MNIST so the model sees
    inputs on the same intensity scale it was trained on.

get_notmnist_transform()
    Pipeline for notMNIST (HuggingFace: anubhavmaity/notMNIST).
    Images arrive as greyscale PIL images at 28x28, so only tensor
    conversion and normalisation are needed.

get_augmented_mnist_transform()
    Optional augmented training transform for the small-data regime.
    Adds mild random affine perturbations realistic for handwritten
    digits without introducing implausible distortions.
    Use only on the training split; val/test always use get_mnist_transform().
"""

from __future__ import annotations

import torch
from torchvision import transforms

# MNIST channel statistics (precomputed over the 60 000 training images).
MNIST_MEAN = (0.1307,)
MNIST_STD  = (0.3081,)


def get_mnist_transform() -> transforms.Compose:
    """Standard in-distribution transform for MNIST, FashionMNIST, and KMNIST."""
    return transforms.Compose([
        transforms.ToTensor(), # → (1, 28, 28) float32 in [0, 1]
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD), # zero-mean, unit-std
    ])


def get_omniglot_transform() -> transforms.Compose:
    """
    Transform for Omniglot.

    Omniglot images are 105x105 RGB PNGs with white strokes on a black
    background — the opposite polarity to MNIST.  This pipeline converts
    to greyscale, resizes to 28x28, inverts pixel values, then normalises
    with MNIST statistics.
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # RGB → L
        transforms.Resize((28, 28)), # 105×105 → 28×28
        transforms.ToTensor(), # → (1, 28, 28) float32 in [0, 1]
        transforms.Lambda(lambda x: 1.0 - x), # invert: white-on-black → black-on-white
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ])


def get_notmnist_transform() -> transforms.Compose:
    """
    Transform for notMNIST (anubhavmaity/notMNIST on HuggingFace Hub).

    Images are already greyscale 28x28 PIL images, so only tensor
    conversion and MNIST normalisation are needed.
    """
    return transforms.Compose([
        transforms.ToTensor(), # → (1, 28, 28) float32 in [0, 1]
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ])


def get_augmented_mnist_transform() -> transforms.Compose:
    """
    Augmented training transform for the small-dataset regime.

    Applies mild random affine perturbations (rotation ±10°, translation
    ±10%, scale 0.9-1.1) before normalisation.  Use only on the training
    split; validation and test splits always use get_mnist_transform().
    """
    return transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MNIST_MEAN, std=MNIST_STD),
    ])