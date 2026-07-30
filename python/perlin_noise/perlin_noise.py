"""Upstream-compatible Perlin noise API backed by Mojo."""

from __future__ import annotations

import random
import math
import numbers
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from ._lib import addr, coordinate_f64, f64, lib
from .rand_vec import RandVec
from .tools import hasher

Coordinate = Union[int, float, List, Tuple]
TileSize = Optional[Union[int, List, Tuple]]


class PerlinNoise:
    """Smooth gradient noise matching ``perlin-noise`` for one to four dimensions."""

    def __init__(self, octaves: float = 1, seed: Optional[int] = None):
        if not isinstance(octaves, (int, float)) or not math.isfinite(octaves):
            raise TypeError("octaves expected to be a finite positive number")
        if octaves <= 0:
            raise ValueError("octaves expected to be positive number")
        if seed is not None and not isinstance(seed, int):
            raise TypeError("seed expected to be an integer")
        if seed is not None and abs(seed) > (1 << 63) - 1:
            raise OverflowError("seed is outside the supported signed 64-bit range")
        self.octaves: float = octaves
        self.seed: int = seed if seed else random.randint(1, 10**5)
        self.cache = {}

    def __call__(
        self,
        coordinates: Coordinate,
        tile_sizes: TileSize = None,
    ) -> float:
        return self.noise(coordinates, tile_sizes)

    def noise(
        self,
        coordinates: Union[int, float, List, Tuple, Iterable],
        tile_sizes: TileSize = None,
    ) -> float:
        if not isinstance(coordinates, (int, float, list, tuple)):
            raise TypeError("coordinates must be int, float or iterable")
        point = (
            [coordinates]
            if isinstance(coordinates, (int, float))
            else coordinates
        )
        dimensions = len(point)
        if not 1 <= dimensions <= 4:
            raise ValueError("Mojo Perlin noise supports one to four dimensions")
        seed = abs(int(self.seed))
        if seed > (1 << 63) - 1:
            raise OverflowError("seed is outside the supported signed 64-bit range")

        tiled = tile_sizes is not None
        if any(
            isinstance(value, numbers.Integral) and abs(value) > 2**53
            for value in point
        ):
            raise OverflowError(
                "integer coordinates must be exactly representable as float64"
            )
        values = [float(value) for value in point]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("coordinates must be finite")
        if not tiled:
            x0 = values[0] * self.octaves
            x1 = values[1] * self.octaves if dimensions > 1 else 0.0
            x2 = values[2] * self.octaves if dimensions > 2 else 0.0
            x3 = values[3] * self.octaves if dimensions > 3 else 0.0
            if not all(math.isfinite(value) for value in (x0, x1, x2, x3)):
                raise OverflowError("scaled coordinates must be finite")
            return float(
                lib().mpn_perlin_scalar(
                    dimensions,
                    seed,
                    0,
                    x0,
                    x1,
                    x2,
                    x3,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                )
            )

        if isinstance(tile_sizes, int):
            tiles = [tile_sizes]
        else:
            tiles = list(tile_sizes)
        if not all(isinstance(tile, int) for tile in tiles):
            raise TypeError("tile_sizes must be int or list of int")
        if len(tiles) != dimensions:
            raise ValueError("tile_sizes must have same length as coordinates")
        if any(tile == 0 for tile in tiles):
            raise ValueError("tile_sizes must be nonzero")
        values = [
            (value % tile) * self.octaves
            for value, tile in zip(values, tiles)
        ]
        periods = [float(tile * self.octaves) for tile in tiles]
        if not all(math.isfinite(value) for value in values + periods):
            raise OverflowError("scaled coordinates and tile periods must be finite")
        values.extend([0.0] * (4 - dimensions))
        periods.extend([1.0] * (4 - dimensions))
        return float(
            lib().mpn_perlin_scalar(
                dimensions,
                seed,
                1,
                *values,
                *periods,
            )
        )

    def noise_array(
        self,
        coordinates,
        tile_sizes: TileSize = None,
        *,
        device: str = "cpu",
    ) -> np.ndarray:
        """Evaluate points with shape ``(..., dimensions)`` in one Mojo call."""
        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        points = coordinate_f64(coordinates)
        scalar_result = points.ndim == 1
        if points.ndim < 1:
            raise TypeError("coordinates must contain at least one dimension")
        dimensions = int(points.shape[-1])
        if not 1 <= dimensions <= 4:
            raise ValueError("Mojo Perlin noise supports one to four dimensions")
        original_shape = points.shape[:-1]
        matrix = np.ascontiguousarray(points.reshape(-1, dimensions))
        if not np.all(np.isfinite(matrix)):
            raise ValueError("coordinates must be finite")
        if matrix.shape[0] == 0:
            return np.empty(original_shape, dtype=np.float64)

        tiled = tile_sizes is not None
        if tiled:
            if isinstance(tile_sizes, int):
                tiles = [tile_sizes]
            else:
                tiles = list(tile_sizes)
            if not all(isinstance(tile, int) for tile in tiles):
                raise TypeError("tile_sizes must be int or list of int")
            if len(tiles) != dimensions:
                raise ValueError("tile_sizes must have same length as coordinates")
            if any(tile == 0 for tile in tiles):
                raise ValueError("tile_sizes must be nonzero")
            tile_array = np.asarray(tiles, dtype=np.float64)
            matrix = np.remainder(matrix, tile_array)
            periods = f64(tile_array * self.octaves)
            matrix *= self.octaves
            scaled = matrix
        else:
            periods = np.ones(dimensions, dtype=np.float64)
            scaled = (
                matrix
                if self.octaves == 1.0
                else f64(matrix * self.octaves)
            )
        if not np.all(np.isfinite(scaled)) or not np.all(np.isfinite(periods)):
            raise OverflowError("scaled coordinates and tile periods must be finite")

        result = np.empty(scaled.shape[0], dtype=np.float64)
        seed = abs(int(self.seed))
        if seed > (1 << 63) - 1:
            raise OverflowError("seed is outside the supported signed 64-bit range")
        use_gpu = device == "gpu"
        if use_gpu and scaled.nbytes + periods.nbytes + result.nbytes >= 2_000_000_000:
            raise ValueError("GPU buffers must total less than 2 GB")
        kernel = (
            lib().mpn_perlin_gpu_batch if use_gpu else lib().mpn_perlin_batch
        )
        status = kernel(
            addr(scaled),
            scaled.shape[0],
            dimensions,
            seed,
            addr(periods),
            int(tiled),
            addr(result),
        )
        if status:
            target = "GPU" if use_gpu else "CPU"
            raise RuntimeError(f"Perlin {target} kernel failed with status {status}")
        reshaped = result.reshape(original_shape)
        return reshaped[()] if scalar_result else reshaped

    @lru_cache(maxsize=100, typed=False)
    def get_from_cache_of_create_new(
        self,
        coors: Tuple[int, ...],
        tile_sizes: Optional[Tuple[int, ...]] = None,
    ) -> RandVec:
        return RandVec(coors, self.seed * hasher(coors, tile_sizes))
