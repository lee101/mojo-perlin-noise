"""Two- and three-dimensional simplex noise backed by Mojo."""

from __future__ import annotations

import math
import numpy as np

from ._lib import addr, coordinate_f64, f64, i64, lib

_CLASSIC_PERMUTATION = (
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3,
    64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85,
    212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43,
    172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
    112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191,
    179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150,
    254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195,
    78, 66, 215, 61, 156, 180,
)


def _permutation(seed: int) -> np.ndarray:
    values = list(_CLASSIC_PERMUTATION)
    if seed:
        state = int(seed) & ((1 << 64) - 1)
        for index in range(255, 0, -1):
            state ^= state >> 12
            state ^= (state << 25) & ((1 << 64) - 1)
            state ^= state >> 27
            state &= (1 << 64) - 1
            choice = (
                state * 2685821657736338717 & ((1 << 64) - 1)
            ) % (index + 1)
            values[index], values[choice] = values[choice], values[index]
    return i64(values + values)


class SimplexNoise:
    """Classic simplex gradient noise with optional fractal octave summation."""

    def __init__(
        self,
        seed: int = 0,
        octaves: int = 1,
        persistence: float = 0.5,
        lacunarity: float = 2.0,
    ):
        if not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        if not isinstance(octaves, int) or octaves < 1:
            raise ValueError("octaves must be a positive integer")
        if octaves > (1 << 63) - 1:
            raise OverflowError("octaves is outside the supported signed 64-bit range")
        if not isinstance(persistence, (int, float)) or not math.isfinite(persistence):
            raise TypeError("persistence must be a finite number")
        if persistence <= 0:
            raise ValueError("persistence must be positive")
        if not isinstance(lacunarity, (int, float)) or not math.isfinite(lacunarity):
            raise TypeError("lacunarity must be a finite number")
        if lacunarity <= 0:
            raise ValueError("lacunarity must be positive")
        self.seed = seed
        self.octaves = octaves
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)
        self._permutation = _permutation(seed)

    def __call__(self, coordinates) -> float:
        return self.noise(coordinates)

    def noise(self, coordinates) -> float:
        point = coordinate_f64(coordinates)
        if point.shape not in ((2,), (3,)):
            raise ValueError("simplex coordinates must have two or three values")
        return float(self.noise_array(point))

    def noise2(self, x: float, y: float) -> float:
        return self.noise((x, y))

    def noise3(self, x: float, y: float, z: float) -> float:
        return self.noise((x, y, z))

    def noise_array(self, coordinates) -> np.ndarray:
        """Evaluate an array whose final axis has length two or three."""
        points = coordinate_f64(coordinates)
        if points.ndim < 1 or points.shape[-1] not in (2, 3):
            raise ValueError(
                "simplex coordinates must end in a two- or three-value axis"
            )
        dimensions = int(points.shape[-1])
        original_shape = points.shape[:-1]
        matrix = f64(points.reshape(-1, dimensions))
        if not np.all(np.isfinite(matrix)):
            raise ValueError("simplex coordinates must be finite")
        if matrix.shape[0] == 0:
            return np.empty(original_shape, dtype=np.float64)
        frequency = amplitude = normalizer = 1.0
        for _ in range(1, self.octaves):
            frequency *= self.lacunarity
            amplitude *= self.persistence
            normalizer += amplitude
        if (
            not math.isfinite(frequency)
            or not math.isfinite(normalizer)
            or not np.all(np.isfinite(matrix * frequency))
        ):
            raise OverflowError("simplex octave scaling must remain finite")
        result = np.empty(matrix.shape[0], dtype=np.float64)
        function = (
            lib().mpn_simplex2_batch
            if dimensions == 2
            else lib().mpn_simplex3_batch
        )
        status = function(
            addr(matrix),
            matrix.shape[0],
            addr(self._permutation),
            self.octaves,
            self.persistence,
            self.lacunarity,
            addr(result),
        )
        if status:
            raise RuntimeError("simplex kernel rejected the input")
        return result.reshape(original_shape)
