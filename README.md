# mojo-perlin-noise

`mojo-perlin-noise` is a Mojo implementation of gradient noise with a small
Python API. It is a drop-in replacement for the commonly used surface of
[`perlin-noise`](https://pypi.org/project/perlin-noise/) 1.14 and adds a
batched NumPy path plus classic simplex noise.

The compute-heavy lattice evaluation runs in a compiled Mojo shared library.
The Python layer handles validation, shapes, and one `ctypes` call per scalar
or batch.

## Coverage

Implemented:

- `PerlinNoise(octaves=1, seed=None)`, `noise(coordinates, tile_sizes=None)`,
  and `__call__`, with numerical parity against `perlin-noise` 1.14
- Perlin noise in one through four dimensions, including negative
  coordinates, floating-point octave scales, deterministic seeds, and tiling
- `PerlinNoise.noise_array(points, tile_sizes=None, device="cpu")` for arrays
  with shape `(..., dimensions)`, with an explicit optional GPU device
- Upstream `RandVec` and helper-module compatibility
- `SimplexNoise` in two and three dimensions, including deterministic
  permutations, batched evaluation, and normalized fractal octave summation

Not implemented:

- Perlin noise above four dimensions; upstream accepts arbitrary dimensions
- Exact compatibility with upstream's accidental acceptance of invalid
  constructor types; this port validates finite numbers and signed 64-bit seeds
- 1D or 4D simplex noise
- derivatives, domain warping, or ridged noise

The upstream `perlin-noise` package contains Perlin noise only. `SimplexNoise`
is an extension in this repository and is checked against an independent
pure-Python implementation of the classic Stefan Gustavson algorithm.

## Install

The checked-in Pixi environment pins the tested Mojo nightly and installs the
real upstream package for parity tests.

```bash
pixi install
pixi run build
```

The build creates `dist/libmojo-perlin-noise.so`. The Pixi environment sets
`PYTHONPATH=python`, so no separate editable install is needed in a checkout.

## Usage

```python
import numpy as np
from perlin_noise import PerlinNoise, SimplexNoise

perlin = PerlinNoise(octaves=2.0, seed=42)
print(perlin([0.1, 0.2]))
# -0.04667200236385356

points = np.array([
    [0.1, 0.2],
    [1.5, -0.25],
    [3.0, 4.0],
])
print(perlin.noise_array(points))
# [-0.046672   -0.25040776  0.        ]

# GPU execution is explicit; unavailable or busy GPUs fall back to CPU.
print(perlin.noise_array(points, device="gpu"))

simplex = SimplexNoise(seed=42, octaves=3)
print(simplex.noise2(0.1, 0.2))
# -0.23117813519779049
print(simplex.noise_array(points))
# [-0.23117814 -0.4522642  -0.50418034]
```

Run the example after `pixi run build` with `pixi run python your_script.py`.

## Benchmarks

Measured on an Intel Xeon E5-2697 v4 at 2.30 GHz with Python 3.12.13. Each
Mojo time is the best of three warm runs. The slower Python reference is run
once. `pixi run bench` holds the repository-wide benchmark lock and verifies
the outputs before printing any result. The GPU row is run only when
`nvidia-smi` reports at least 4,000 MiB free.

| Kernel | Samples | Mojo | Reference | Speedup | Compared with |
|---|---:|---:|---:|---:|---|
| Perlin 2D scalar calls | 1,000 | 33.26 ms | 148.45 ms | 4.46x | perlin-noise 1.14 |
| Perlin 2D batch | 10,000 | 33.25 ms | 1551.36 ms | 46.65x | perlin-noise 1.14 |
| Perlin 2D batch, GPU | 10,000 | 1.45 ms | 33.25 ms | 22.96x | Mojo CPU |
| Perlin 3D batch | 5,000 | 47.60 ms | 1657.28 ms | 34.81x | perlin-noise 1.14 |
| Perlin 4D batch | 2,000 | 48.28 ms | 1378.19 ms | 28.55x | perlin-noise 1.14 |
| Simplex 2D, 3 octaves | 25,000 | 5.36 ms | 589.99 ms | 110.00x | pure Python |
| Simplex 3D, 3 octaves | 15,000 | 5.15 ms | 548.62 ms | 106.48x | pure Python |

The Perlin and simplex reference rows compare a Mojo call with the scalar-only
upstream Perlin API or a scalar pure-Python simplex loop. The GPU row compares
the explicit GPU path with this library's default CPU path.

Reproduce them with:

```bash
pixi run bench
```

## How it works

`src/noise.mojo` is one compilation unit. Its non-parametric exported
functions use `@export` and the C ABI. Python passes C-contiguous NumPy buffers
as integer addresses because pointer origins cannot cross this ABI. Coordinates
are row-major `float64` arrays, simplex permutations are `int64`, and Mojo
writes directly into a preallocated `float64` result. The wrapper validates
shapes, exact large-integer conversion, finite values, dtype, contiguity, and
buffer sizes before the synchronous call; empty arrays never become pointers,
and local references keep every NumPy buffer alive until the call returns.

Scalar Perlin calls use a direct scalar ABI and avoid constructing temporary
NumPy input and output arrays. C-contiguous batch inputs with `octaves=1` cross
the FFI boundary without a coordinate copy.

Upstream Perlin gradients are unusual: every lattice vertex reseeds Python's
Mersenne Twister and draws one uniform value per dimension. The Mojo kernel
reproduces CPython's integer seeding, MT19937 twist, and 53-bit `random()` path,
then applies the upstream fade and coordinate-hash rules. The independent
lanes of the MT19937 twist use native-width SIMD with scalar remainder loops.
Per-axis lattice, distance, and fade setup uses float64 SIMD with a scalar tail,
and the common one-word MT seed avoids generic multi-word branches. Large CPU
batches are split into 256-row tasks across at most 16 workers; batches below
1,024 rows remain serial. Seeded results still match the Python package to
floating-point rounding rather than merely producing a similar-looking field.

Perlin's arithmetic-heavy batch kernel also has an explicit `device="gpu"`
path. It is attempted only when `nvidia-smi` reports at least 4,000 MiB free.
Device buffers are scoped to one call and capped below 2 GB. Missing devices,
low free memory, and context, allocation, or launch failures fall back to CPU;
CPU remains the default.

Simplex uses the classic 256-entry permutation and simplex-corner attenuation
in 2D or 3D. A seeded xorshift64* Fisher-Yates shuffle creates stable alternate
permutations. Multiple octaves are accumulated in the same Mojo call and
normalized by their total amplitude.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The test suite compares Mojo directly with the installed `perlin-noise` 1.14
package and checks simplex against the independent reference over seeded,
negative-coordinate, tiled, batched, and fractal cases.

## License

MIT
