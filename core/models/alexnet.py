"""
core/models/alexnet.py
──────────────────────
AlexNet-inspired convolutional classifier scaled for 28x28 greyscale inputs.

A single class serves both inference modes, matching the LeNet5 interface:
  - Deterministic  →  model.eval()                     (dropout disabled)
  - MC Dropout     →  model.eval() + enable_dropout()  (see MCDropoutPredictor)

Input pre-processing
--------------------
MNIST images are single-channel (1, 28, 28).  Replicate the channel to
three before passing to this model, e.g.:
    x3 = x.repeat(1, 3, 1, 1)   # (B, 1, 28, 28) → (B, 3, 28, 28)

Architecture
------------
Input : (B, 3, 28, 28)  — 3-channel, 28x28

Feature extractor  (5 conv blocks with BatchNorm + ReLU)
  Block 1: Conv2d(  3,  64, 3, p=1) → BN → ReLU → MaxPool(2)  →  (B,  64, 14, 14)
  Block 2: Conv2d( 64, 192, 3, p=1) → BN → ReLU               →  (B, 192, 14, 14)
  Block 3: Conv2d(192, 384, 3, p=1) → BN → ReLU               →  (B, 384, 14, 14)
  Block 4: Conv2d(384, 256, 3, p=1) → BN → ReLU               →  (B, 256, 14, 14)
  Block 5: Conv2d(256, 256, 3, p=1) → BN → ReLU → MaxPool(2)  →  (B, 256,  7,  7)

Classifier
  Dropout(p)
  Flatten                           →  (B, 12544)
  Linear(12544, 1024) → ReLU
  Dropout(p)
  Linear( 1024,  512) → ReLU
  Linear(  512,  n_classes)         →  (B, n_classes)  logits

Output: raw logits (cross-entropy loss handles softmax internally).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AlexNetSmall(nn.Module):
    """
    AlexNet-inspired architecture scaled for 28x28 inputs.

    Parameters
    ----------
    n_classes : int
        Number of output classes.  Default: 10 (MNIST).
    dropout_rate : float
        Bernoulli dropout probability applied after the feature extractor
        and between the fully-connected hidden layers.
        Set to 0.0 to disable dropout entirely.

    Examples
    --------
    >>> model = AlexNetSmall(dropout_rate=0.5)
    >>> x = torch.randn(4, 3, 28, 28)
    >>> logits = model(x)          # (4, 10)
    >>> logits.shape
    torch.Size([4, 10])

    MC Dropout inference
    --------------------
    >>> model.eval()
    >>> model.enable_dropout()     # re-enables nn.Dropout layers only
    >>> # repeated forward passes now produce stochastic outputs
    """

    def __init__(
        self,
        n_classes: int = 10,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                f"dropout_rate must be in [0, 1), got {dropout_rate}"
            )

        self.dropout_rate = dropout_rate

        # ── Feature extractor (5 conv blocks) ─────────────────────────────────
        # No dropout in the convolutional block — BatchNorm provides
        # regularisation here; dropout on feature maps is applied separately.
        self.features = nn.Sequential(
            # Block 1  — 28×28 → 14×14
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2  — 14×14 → 14×14
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            # Block 3  — 14×14 → 14×14
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            # Block 4  — 14×14 → 14×14
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Block 5  — 14×14 → 7×7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Classifier with dropout ────────────────────────────────────────────
        # nn.Dropout modules (not functional) so that enable_dropout() can
        # selectively restore stochastic behaviour after model.eval().
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Flatten(),

            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(512, n_classes)

        self._init_weights()

    # ── Weight initialisation ──────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Kaiming normal for Conv2d/Linear; ones/zeros for BatchNorm."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 28, 28)
            3-channel 28x28 image batch.  For greyscale MNIST inputs,
            replicate the channel dimension before calling:
                x = x.repeat(1, 3, 1, 1)

        Returns
        -------
        torch.Tensor, shape (B, n_classes)
            Raw class logits (no softmax applied).
        """
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    # ── Convenience ────────────────────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the total (or trainable-only) parameter count."""
        params = (
            self.parameters() if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)

    def enable_dropout(self) -> None:
        """
        Set all Dropout layers back to training mode while leaving the rest
        of the model in eval mode.

        Call this after model.eval() to enable MC Dropout inference.
        BatchNorm layers remain in eval mode so their running statistics
        are not updated during inference.
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def __repr__(self) -> str:
        return (
            f"AlexNetSmall("
            f"dropout_rate={self.dropout_rate}, "
            f"params={self.num_parameters():,})"
        )