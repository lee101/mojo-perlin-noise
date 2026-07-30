"""Numerical and behavioral parity tests."""

import json
import os
import subprocess
import sys
import tempfile
import numpy as np
import pytest

from perlin_noise import PerlinNoise, SimplexNoise
from perlin_noise import perlin_noise as perlin_module
from perlin_noise.tools import dot, fade, hasher, product, sample_vector
from reference_simplex import fractal


def upstream_values(cases):
    script = """
import json, sys
from perlin_noise import PerlinNoise
cases = json.loads(sys.stdin.read())
values = []
for case in cases:
    noise = PerlinNoise(octaves=case["octaves"], seed=case["seed"])
    values.append(noise(case["coordinates"], case.get("tile_sizes")))
print(json.dumps(values))
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory() as directory:
        process = subprocess.run(
            [sys.executable, "-c", script],
            input=json.dumps(cases),
            text=True,
            capture_output=True,
            check=True,
            cwd=directory,
            env=environment,
        )
    return np.asarray(json.loads(process.stdout))


@pytest.mark.parametrize("dimensions", [1, 2, 3, 4])
def test_perlin_scalar_matches_real_upstream(dimensions):
    cases = []
    for seed in (1, 42, 987654):
        for octaves in (0.5, 1, 2.75):
            coordinates = [
                (-1 if axis % 2 else 1) * (0.173 + axis * 1.237)
                for axis in range(dimensions)
            ]
            cases.append(
                {"coordinates": coordinates, "octaves": octaves, "seed": seed}
            )
    expected = upstream_values(cases)
    actual = [
        PerlinNoise(octaves=case["octaves"], seed=case["seed"])(
            case["coordinates"]
        )
        for case in cases
    ]
    assert np.allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_perlin_tiling_matches_real_upstream():
    cases = [
        {
            "coordinates": [3.1, -2.2],
            "tile_sizes": [3, 5],
            "octaves": 2.5,
            "seed": 42,
        },
        {
            "coordinates": [-7.25, 12.75, 1.125],
            "tile_sizes": [4, 7, 3],
            "octaves": 1.5,
            "seed": 137,
        },
    ]
    expected = upstream_values(cases)
    actual = [
        PerlinNoise(octaves=case["octaves"], seed=case["seed"])(
            case["coordinates"], case["tile_sizes"]
        )
        for case in cases
    ]
    assert np.allclose(actual, expected, rtol=2e-15, atol=2e-15)


def test_perlin_batch_matches_real_upstream():
    rng = np.random.default_rng(7)
    points = rng.uniform(-8, 8, size=(120, 3))
    cases = [
        {"coordinates": point.tolist(), "octaves": 1.75, "seed": 777}
        for point in points
    ]
    expected = upstream_values(cases)
    actual = PerlinNoise(octaves=1.75, seed=777).noise_array(points)
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_perlin_simd_tail_matches_real_upstream():
    points = np.array(
        [
            [-7.125, 0.25, 3.75],
            [1.0 / 3.0, -2.5, 19.125],
            [123.5, -456.25, 789.75],
        ]
    )
    cases = [
        {"coordinates": point.tolist(), "octaves": 1.25, "seed": 314159}
        for point in points
    ]
    expected = upstream_values(cases)
    actual = PerlinNoise(octaves=1.25, seed=314159).noise_array(points)
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_perlin_gpu_path_or_cpu_fallback():
    rng = np.random.default_rng(17)
    points = rng.uniform(-20, 20, size=(129, 2))
    noise = PerlinNoise(octaves=2.5, seed=91)
    expected = noise.noise_array(points)
    actual = noise.noise_array(points, device="gpu")
    assert np.allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_perlin_gpu_failure_is_reported(monkeypatch):
    points = np.array([[0.125, -0.75], [4.5, 2.25]])
    noise = PerlinNoise(octaves=1.5, seed=71)
    class FakeLibrary:
        mpn_perlin_gpu_batch = staticmethod(lambda *_: 2)

    fake_library = FakeLibrary()
    monkeypatch.setattr(perlin_module, "lib", lambda: fake_library)
    with pytest.raises(RuntimeError, match="GPU kernel failed with status 2"):
        noise.noise_array(points, device="gpu")


def test_perlin_large_seed_product_matches_real_upstream():
    case = {
        "coordinates": [12345.25, -2345.5, 345.75, -45.125],
        "octaves": 1.5,
        "seed": (1 << 62) + 57,
    }
    expected = upstream_values([case])[0]
    actual = PerlinNoise(octaves=case["octaves"], seed=case["seed"])(
        case["coordinates"]
    )
    assert actual == pytest.approx(expected, rel=2e-15, abs=2e-15)


def test_perlin_batch_shape_and_scalar_api():
    points = np.zeros((5, 7, 2))
    values = PerlinNoise(seed=11).noise_array(points)
    assert values.shape == (5, 7)
    assert np.all(values == 0)
    assert isinstance(PerlinNoise(seed=11)([0.1, 0.2]), float)
    assert PerlinNoise(seed=11).noise(0.25) == pytest.approx(
        PerlinNoise(seed=11)([0.25])
    )
    assert PerlinNoise(seed=11).noise_array(np.empty((0, 2))).shape == (0,)


def test_perlin_tiling_is_periodic():
    noise = PerlinNoise(octaves=2, seed=23)
    point = np.array([0.375, -1.25])
    base = noise(point.tolist(), [3, 5])
    assert noise((point + [3, 0]).tolist(), [3, 5]) == pytest.approx(base)
    assert noise((point + [0, 5]).tolist(), [3, 5]) == pytest.approx(base)


def test_perlin_validation_matches_upstream_contract():
    with pytest.raises(ValueError, match="octaves"):
        PerlinNoise(octaves=0)
    with pytest.raises(TypeError, match="coordinates"):
        PerlinNoise(seed=1).noise(np.array([1.0, 2.0]))
    with pytest.raises(TypeError, match="tile_sizes"):
        PerlinNoise(seed=1)([1.0, 2.0], [2.0, 3])
    with pytest.raises(ValueError, match="same length"):
        PerlinNoise(seed=1)([1.0, 2.0], [3])
    with pytest.raises(ValueError, match="one to four"):
        PerlinNoise(seed=1)([0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="device"):
        PerlinNoise(seed=1).noise_array([[0.1, 0.2]], device="tpu")
    with pytest.raises(ValueError, match="finite"):
        PerlinNoise(seed=1).noise_array([[np.nan, 0.2]])
    with pytest.raises(ValueError, match="nonzero"):
        PerlinNoise(seed=1)([0.1, 0.2], [0, 2])
    with pytest.raises(TypeError, match="seed"):
        PerlinNoise(seed=1.5)
    with pytest.raises(OverflowError, match="exactly representable"):
        PerlinNoise(seed=1).noise_array([[2**53 + 1, 0]])


def test_upstream_randvec_compatibility_surface():
    noise = PerlinNoise(seed=17)
    vector = noise.get_from_cache_of_create_new((2, -3))
    assert vector is noise.get_from_cache_of_create_new((2, -3))
    assert vector.coordinates == (2, -3)
    assert vector.vec == pytest.approx(
        [0.9185621835159932, 0.5372018784939248]
    )
    assert vector.get_weighted_val([2.25, -2.5]) == pytest.approx(
        vector.weight_to([2.25, -2.5])
        * sum(
            value * distance
            for value, distance in zip(vector.vec, vector.dists_to([2.25, -2.5]))
        )
    )


def test_upstream_helper_module_compatibility_surface():
    assert dot([1, 2], (3, 4)) == 11
    with pytest.raises(ValueError, match="lengths"):
        dot([1], [1, 2])
    assert fade(0.25) == pytest.approx(0.103515625)
    assert hasher((2, -3)) == 27
    assert product([2, 3, 4]) == 24
    import random

    state = random.getstate()
    assert sample_vector(2, 17) == pytest.approx(
        [0.0439678194249864, 0.6133815542373582]
    )
    assert random.getstate() == state


@pytest.mark.parametrize("dimensions", [2, 3])
@pytest.mark.parametrize("seed", [0, 1, 123456])
def test_simplex_matches_independent_reference(dimensions, seed):
    rng = np.random.default_rng(seed)
    points = rng.uniform(-10, 10, size=(100, dimensions))
    noise = SimplexNoise(
        seed=seed, octaves=3, persistence=0.6, lacunarity=1.9
    )
    expected = np.array(
        [
            fractal(
                point,
                noise._permutation,
                noise.octaves,
                noise.persistence,
                noise.lacunarity,
            )
            for point in points
        ]
    )
    assert np.allclose(noise.noise_array(points), expected, atol=2e-14, rtol=2e-14)


def test_simplex_classic_permutation_vectors():
    noise = SimplexNoise(seed=0)
    points2 = np.array([[0, 0], [0.1, 0.2], [-1.25, 2.5], [10.75, -4.125]])
    points3 = np.array(
        [[0, 0, 0], [0.1, 0.2, 0.3], [-1.25, 2.5, 0.75], [8, -3, 1.5]]
    )
    expected2 = [fractal(point, noise._permutation, 1, 0.5, 2) for point in points2]
    expected3 = [fractal(point, noise._permutation, 1, 0.5, 2) for point in points3]
    assert noise.noise_array(points2) == pytest.approx(expected2, abs=1e-14)
    assert noise.noise_array(points3) == pytest.approx(expected3, abs=1e-14)


def test_simplex_scalar_methods_and_determinism():
    first = SimplexNoise(seed=99)
    second = SimplexNoise(seed=99)
    assert first.noise2(0.1, 0.2) == first([0.1, 0.2])
    assert first.noise3(0.1, 0.2, 0.3) == first([0.1, 0.2, 0.3])
    points = np.arange(60, dtype=float).reshape(20, 3) / 7
    assert np.array_equal(first.noise_array(points), second.noise_array(points))


def test_simplex_validation():
    with pytest.raises(ValueError, match="octaves"):
        SimplexNoise(octaves=0)
    with pytest.raises(ValueError, match="two or three"):
        SimplexNoise().noise([1])
    with pytest.raises(ValueError, match="two- or three"):
        SimplexNoise().noise_array(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="finite"):
        SimplexNoise().noise_array([[np.inf, 0]])
    assert SimplexNoise().noise_array(np.empty((0, 3))).shape == (0,)


def test_ffi_rejects_null_nonempty_buffers_and_accepts_empty_batches():
    library = perlin_module.lib()
    assert library.mpn_perlin_batch(0, 1, 2, 1, 0, 0, 0) != 0
    assert library.mpn_perlin_batch(0, 0, 2, 1, 0, 0, 0) == 0
    assert library.mpn_simplex2_batch(0, 1, 0, 1, 0.5, 2.0, 0) != 0
    assert library.mpn_simplex2_batch(0, 0, 0, 1, 0.5, 2.0, 0) == 0
