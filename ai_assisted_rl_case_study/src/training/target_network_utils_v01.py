"""Target-network utility helpers v0.1."""

from __future__ import annotations

import numpy as np


def hard_update_target_network_v01(
    source_network,
    target_network,
) -> None:
    """Copy all source network weights into the target network."""

    source_weights = source_network.get_weights()
    target_weights = target_network.get_weights()
    _validate_weight_compatibility(source_weights, target_weights)
    target_network.set_weights(source_weights)


def compare_network_weights_v01(source_network, target_network) -> bool:
    """Return True when source and target weights have identical values."""

    source_weights = source_network.get_weights()
    target_weights = target_network.get_weights()
    if len(source_weights) != len(target_weights):
        return False
    for source, target in zip(source_weights, target_weights):
        if source.shape != target.shape:
            return False
        if not np.array_equal(source, target):
            return False
    return True


def _validate_weight_compatibility(source_weights, target_weights) -> None:
    if len(source_weights) != len(target_weights):
        raise ValueError(
            "source_network and target_network must have the same number of weights"
        )
    for index, (source, target) in enumerate(zip(source_weights, target_weights)):
        if source.shape != target.shape:
            raise ValueError(
                "source_network and target_network weight shapes differ at "
                f"index {index}: {source.shape} != {target.shape}"
            )
