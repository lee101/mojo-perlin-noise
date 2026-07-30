"""Compatibility ``RandVec`` used by the upstream implementation."""

from __future__ import annotations

import math

from .tools import dot, fade, sample_vector


class RandVec:
    def __init__(self, coordinates, seed: int):
        self.coordinates = coordinates
        self.vec = sample_vector(dimensions=len(coordinates), seed=seed)

    def dists_to(self, coordinates):
        return tuple(
            first - second
            for first, second in zip(coordinates, self.coordinates)
        )

    def weight_to(self, coordinates) -> float:
        return math.prod(
            fade(1 - abs(distance)) for distance in self.dists_to(coordinates)
        )

    def get_weighted_val(self, coordinates) -> float:
        return self.weight_to(coordinates) * dot(
            self.vec,
            self.dists_to(coordinates),
        )
