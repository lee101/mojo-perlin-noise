"""Compatibility helpers from the upstream package."""

from __future__ import annotations

import math
import random


def dot(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ValueError("lengths of two vectors are not equal")
    return sum(val1 * val2 for val1, val2 in zip(vec1, vec2))


def sample_vector(dimensions: int, seed: int):
    state = random.getstate()
    random.seed(seed)
    vector = [random.uniform(-1, 1) for _ in range(dimensions)]
    random.setstate(state)
    return vector


def fade(given_value: float) -> float:
    if given_value < -0.1 or given_value > 1.1:
        raise ValueError("expected to have value in [-0.1, 1.1]")
    return (
        6 * math.pow(given_value, 5)
        - 15 * math.pow(given_value, 4)
        + 10 * math.pow(given_value, 3)
    )


def hasher(coordinates, tile_sizes=None) -> int:
    if tile_sizes:
        coordinates = tuple(
            coordinate % tile for coordinate, tile in zip(coordinates, tile_sizes)
        )
    return max(
        1,
        int(
            abs(
                dot(
                    [10**coordinate for coordinate in range(len(coordinates))],
                    coordinates,
                )
                + 1
            )
        ),
    )


def product(iterable):
    if len(iterable) == 1:
        return iterable[0]
    return iterable[0] * product(iterable[1:])
