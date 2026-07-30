"""Small independent reference implementation of classic simplex noise."""

import math

GRADIENTS = (
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)


def simplex2(x, y, permutation):
    f2 = (math.sqrt(3.0) - 1.0) / 2.0
    g2 = (3.0 - math.sqrt(3.0)) / 6.0
    skew = (x + y) * f2
    i, j = math.floor(x + skew), math.floor(y + skew)
    unskew = (i + j) * g2
    x0, y0 = x - (i - unskew), y - (j - unskew)
    i1, j1 = ((1, 0) if x0 > y0 else (0, 1))
    x1, y1 = x0 - i1 + g2, y0 - j1 + g2
    x2, y2 = x0 - 1 + 2 * g2, y0 - 1 + 2 * g2
    ii, jj = i & 255, j & 255
    indices = (
        permutation[ii + permutation[jj]] % 12,
        permutation[ii + i1 + permutation[jj + j1]] % 12,
        permutation[ii + 1 + permutation[jj + 1]] % 12,
    )
    value = 0.0
    for px, py, gradient in (
        (x0, y0, indices[0]),
        (x1, y1, indices[1]),
        (x2, y2, indices[2]),
    ):
        attenuation = 0.5 - px * px - py * py
        if attenuation > 0:
            gx, gy, _ = GRADIENTS[gradient]
            value += attenuation**4 * (gx * px + gy * py)
    return 70.0 * value


def simplex3(x, y, z, permutation):
    skew = (x + y + z) / 3.0
    i, j, k = math.floor(x + skew), math.floor(y + skew), math.floor(z + skew)
    unskew = (i + j + k) / 6.0
    x0, y0, z0 = x - (i - unskew), y - (j - unskew), z - (k - unskew)
    if x0 >= y0:
        if y0 >= z0:
            first, second = (1, 0, 0), (1, 1, 0)
        elif x0 >= z0:
            first, second = (1, 0, 0), (1, 0, 1)
        else:
            first, second = (0, 0, 1), (1, 0, 1)
    elif y0 < z0:
        first, second = (0, 0, 1), (0, 1, 1)
    elif x0 < z0:
        first, second = (0, 1, 0), (0, 1, 1)
    else:
        first, second = (0, 1, 0), (1, 1, 0)
    offsets = (
        (x0, y0, z0),
        (x0 - first[0] + 1 / 6, y0 - first[1] + 1 / 6, z0 - first[2] + 1 / 6),
        (x0 - second[0] + 1 / 3, y0 - second[1] + 1 / 3, z0 - second[2] + 1 / 3),
        (x0 - 0.5, y0 - 0.5, z0 - 0.5),
    )
    ii, jj, kk = i & 255, j & 255, k & 255
    indices = (
        permutation[ii + permutation[jj + permutation[kk]]] % 12,
        permutation[
            ii + first[0] + permutation[
                jj + first[1] + permutation[kk + first[2]]
            ]
        ] % 12,
        permutation[
            ii + second[0] + permutation[
                jj + second[1] + permutation[kk + second[2]]
            ]
        ] % 12,
        permutation[ii + 1 + permutation[jj + 1 + permutation[kk + 1]]] % 12,
    )
    value = 0.0
    for (px, py, pz), gradient in zip(offsets, indices):
        attenuation = 0.6 - px * px - py * py - pz * pz
        if attenuation > 0:
            gx, gy, gz = GRADIENTS[gradient]
            value += attenuation**4 * (gx * px + gy * py + gz * pz)
    return 32.0 * value


def fractal(point, permutation, octaves, persistence, lacunarity):
    function = simplex2 if len(point) == 2 else simplex3
    value = normalizer = 0.0
    amplitude = frequency = 1.0
    for _ in range(octaves):
        value += amplitude * function(
            *(coordinate * frequency for coordinate in point), permutation
        )
        normalizer += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return value / normalizer
