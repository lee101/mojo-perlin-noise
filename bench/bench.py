"""Benchmark Mojo kernels against upstream and independent Python references."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_DIR = os.path.join(ROOT, "python")
TESTS_DIR = os.path.join(ROOT, "tests")


def load_upstream_perlin():
    original_path = sys.path[:]
    sys.path = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) != os.path.abspath(PYTHON_DIR)
    ]
    for name in list(sys.modules):
        if name == "perlin_noise" or name.startswith("perlin_noise."):
            del sys.modules[name]
    try:
        return importlib.import_module("perlin_noise").PerlinNoise
    finally:
        for name in list(sys.modules):
            if name == "perlin_noise" or name.startswith("perlin_noise."):
                del sys.modules[name]
        sys.path = original_path


UpstreamPerlinNoise = load_upstream_perlin()
from perlin_noise import PerlinNoise, SimplexNoise  # noqa: E402

sys.path.insert(0, TESTS_DIR)
from reference_simplex import fractal  # noqa: E402


def timed(function, repetitions=3):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown CPU"


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    return f"{seconds * 1e3:.2f} ms"


def gpu_memory_free_mib():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return min(int(line.strip()) for line in output.splitlines())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def main():
    rng = np.random.default_rng(20260730)
    rows = []
    gpu_free_mib = gpu_memory_free_mib()

    points = rng.uniform(-100, 100, size=(1_000, 2))
    mojo_noise = PerlinNoise(octaves=2.5, seed=42)
    upstream_noise = UpstreamPerlinNoise(octaves=2.5, seed=42)
    mojo_noise(points[0].tolist())
    mojo_seconds, mojo_values = timed(
        lambda: np.array([mojo_noise(point.tolist()) for point in points]),
        repetitions=3,
    )
    reference_seconds, reference_values = timed(
        lambda: np.array([upstream_noise(point.tolist()) for point in points]),
        repetitions=1,
    )
    if not np.allclose(mojo_values, reference_values, rtol=2e-14, atol=2e-14):
        raise RuntimeError("Perlin scalar benchmark outputs do not match upstream")
    rows.append(
        (
            "Perlin 2D scalar calls",
            f"{len(points):,}",
            mojo_seconds,
            reference_seconds,
            "perlin-noise 1.14",
        )
    )

    for dimensions, count in ((2, 10_000), (3, 5_000), (4, 2_000)):
        points = rng.uniform(-100, 100, size=(count, dimensions))
        mojo_noise = PerlinNoise(octaves=2.5, seed=42)
        upstream_noise = UpstreamPerlinNoise(octaves=2.5, seed=42)
        mojo_noise.noise_array(points[:8])
        mojo_seconds, mojo_values = timed(
            lambda: mojo_noise.noise_array(points), repetitions=3
        )
        reference_seconds, reference_values = timed(
            lambda: np.array([upstream_noise(point.tolist()) for point in points]),
            repetitions=1,
        )
        if not np.allclose(mojo_values, reference_values, rtol=2e-14, atol=2e-14):
            raise RuntimeError("Perlin benchmark outputs do not match upstream")
        rows.append(
            (
                f"Perlin {dimensions}D batch",
                f"{count:,}",
                mojo_seconds,
                reference_seconds,
                "perlin-noise 1.14",
            )
        )
        if dimensions == 2 and gpu_free_mib is not None and gpu_free_mib >= 4_000:
            mojo_noise.noise_array(points[:128], device="gpu")
            gpu_seconds, gpu_values = timed(
                lambda: mojo_noise.noise_array(points, device="gpu"),
                repetitions=3,
            )
            if not np.allclose(
                gpu_values, mojo_values, rtol=2e-14, atol=2e-14
            ):
                raise RuntimeError("Perlin GPU benchmark outputs do not match CPU")
            rows.append(
                (
                    "Perlin 2D batch, GPU",
                    f"{count:,}",
                    gpu_seconds,
                    mojo_seconds,
                    "Mojo CPU",
                )
            )

    for dimensions, count in ((2, 25_000), (3, 15_000)):
        points = rng.uniform(-100, 100, size=(count, dimensions))
        mojo_noise = SimplexNoise(
            seed=42, octaves=3, persistence=0.6, lacunarity=1.9
        )
        mojo_noise.noise_array(points[:8])
        mojo_seconds, mojo_values = timed(
            lambda: mojo_noise.noise_array(points), repetitions=3
        )
        reference_seconds, reference_values = timed(
            lambda: np.array(
                [
                    fractal(
                        point,
                        mojo_noise._permutation,
                        mojo_noise.octaves,
                        mojo_noise.persistence,
                        mojo_noise.lacunarity,
                    )
                    for point in points
                ]
            ),
            repetitions=1,
        )
        if not np.allclose(mojo_values, reference_values, rtol=2e-13, atol=2e-13):
            raise RuntimeError("simplex benchmark outputs do not match reference")
        rows.append(
            (
                f"Simplex {dimensions}D, 3 octaves",
                f"{count:,}",
                mojo_seconds,
                reference_seconds,
                "pure Python",
            )
        )

    print(f"Machine: {cpu_name()}, Python {sys.version.split()[0]}")
    if gpu_free_mib is None:
        print("GPU benchmark skipped: free device memory could not be checked")
    elif gpu_free_mib < 4_000:
        print(
            f"GPU benchmark skipped: {gpu_free_mib} MiB free is below 4,000 MiB"
        )
    print()
    print("| Kernel | Samples | Mojo | Reference | Speedup | Compared with |")
    print("|---|---:|---:|---:|---:|---|")
    for name, count, mojo_seconds, reference_seconds, reference_name in rows:
        print(
            f"| {name} | {count} | {format_time(mojo_seconds)} | "
            f"{format_time(reference_seconds)} | "
            f"{reference_seconds / mojo_seconds:.2f}x | {reference_name} |"
        )


if __name__ == "__main__":
    main()
