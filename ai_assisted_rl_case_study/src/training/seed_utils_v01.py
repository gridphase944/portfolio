"""Seed helpers for training smoke utilities v0.1."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_global_seed_v01(seed: int) -> None:
    """Set Python, NumPy, and TensorFlow random seeds for smoke reproducibility."""

    normalized_seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(normalized_seed)
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    tf.random.set_seed(normalized_seed)
