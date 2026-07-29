from __future__ import annotations

import numpy as np

from dopa_api.visualization import normalize_response_magnitude


def test_normalizes_absolute_cortical_response() -> None:
    predictions = np.array([[-2.0, 0.0, 1.0], [4.0, -1.0, 0.0]])

    normalized = normalize_response_magnitude(predictions)

    assert normalized.shape == predictions.shape
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert normalized[0, 0] > normalized[0, 2]


def test_flat_cortical_response_normalizes_to_zero() -> None:
    normalized = normalize_response_magnitude(np.zeros((2, 20_484)))

    assert not normalized.any()
