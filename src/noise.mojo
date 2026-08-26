"""Perlin and simplex noise kernels exposed through a small C ABI."""

from std.math import floor
from max.algorithm import parallelize
from std.gpu import global_idx
from max.gpu.host import DeviceContext
from std.sys.info import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]
comptime PERLIN_PARALLEL_THRESHOLD = 1_024
comptime PERLIN_PARALLEL_GRAIN = 256
comptime PERLIN_PARALLEL_WORKERS = 16


def mt_twist[state_origin: MutOrigin](
    state: UnsafePointer[UInt32, state_origin]
):
    comptime matrix_a = UInt32(0x9908B0DF)
    comptime upper = UInt32(0x80000000)
    comptime lower = UInt32(0x7FFFFFFF)
    comptime W = simd_width_of[DType.uint32]()
    var i = 0
    while i + W <= 227:
        var word_vec_first = (
            state.load[width=W](i) & upper
        ) | (state.load[width=W](i + 1) & lower)
        state.store(
            i,
            state.load[width=W](i + 397)
            ^ (word_vec_first >> UInt32(1))
            ^ ((word_vec_first & UInt32(1)) * matrix_a),
        )
        i += W
    while i < 227:
        var word_tail_first = (state[i] & upper) | (state[i + 1] & lower)
        state[i] = (
            state[i + 397]
            ^ (word_tail_first >> UInt32(1))
            ^ ((word_tail_first & UInt32(1)) * matrix_a)
        )
        i += 1
    i = 227
    while i + W <= 623:
        var word_vec_second = (
            state.load[width=W](i) & upper
        ) | (state.load[width=W](i + 1) & lower)
        state.store(
            i,
            state.load[width=W](i - 227)
            ^ (word_vec_second >> UInt32(1))
            ^ ((word_vec_second & UInt32(1)) * matrix_a),
        )
        i += W
    while i < 623:
        var word_tail_second = (state[i] & upper) | (state[i + 1] & lower)
        state[i] = (
            state[i - 227]
            ^ (word_tail_second >> UInt32(1))
            ^ ((word_tail_second & UInt32(1)) * matrix_a)
        )
        i += 1
    var final_word = (state[623] & upper) | (state[0] & lower)
    state[623] = (
        state[396]
        ^ (final_word >> UInt32(1))
        ^ ((final_word & UInt32(1)) * matrix_a)
    )


def mt_seed(mut state: Array[UInt32, 624], seed: UInt128):
    var pointer = state.unsafe_ptr()
    pointer[0] = UInt32(19650218)
    for i in range(1, 624):
        var previous = pointer[i - 1]
        pointer[i] = (
            UInt32(1812433253) * (previous ^ (previous >> UInt32(30)))
            + UInt32(i)
        )

    var key0 = UInt32(seed)
    var key1 = UInt32(seed >> 32)
    var key2 = UInt32(seed >> 64)
    var key3 = UInt32(seed >> 96)
    var key_length = 1
    if key3 != UInt32(0):
        key_length = 4
    elif key2 != UInt32(0):
        key_length = 3
    elif key1 != UInt32(0):
        key_length = 2
    var i = 1
    var j = 0
    var count = 624
    while count > 0:
        var previous = pointer[i - 1]
        var key = key0
        if j == 1:
            key = key1
        elif j == 2:
            key = key2
        elif j == 3:
            key = key3
        pointer[i] = (
            (pointer[i] ^ (
                (previous ^ (previous >> UInt32(30))) * UInt32(1664525)
            ))
            + key
            + UInt32(j)
        )
        i += 1
        j += 1
        if i >= 624:
            pointer[0] = pointer[623]
            i = 1
        if j >= key_length:
            j = 0
        count -= 1
    count = 623
    while count > 0:
        var previous = pointer[i - 1]
        pointer[i] = (
            (pointer[i] ^ (
                (previous ^ (previous >> UInt32(30))) * UInt32(1566083941)
            ))
            - UInt32(i)
        )
        i += 1
        if i >= 624:
            pointer[0] = pointer[623]
            i = 1
        count -= 1
    pointer[0] = UInt32(0x80000000)
    mt_twist(pointer)


def mt_seed_one_word(mut state: Array[UInt32, 624], seed: UInt32):
    var pointer = state.unsafe_ptr()
    pointer[0] = UInt32(19650218)
    for i in range(1, 624):
        var previous = pointer[i - 1]
        pointer[i] = (
            UInt32(1812433253) * (previous ^ (previous >> UInt32(30)))
            + UInt32(i)
        )
    var i = 1
    var count = 624
    while count > 0:
        var previous = pointer[i - 1]
        pointer[i] = (
            (pointer[i] ^ (
                (previous ^ (previous >> UInt32(30))) * UInt32(1664525)
            ))
            + seed
        )
        i += 1
        if i >= 624:
            pointer[0] = pointer[623]
            i = 1
        count -= 1
    count = 623
    while count > 0:
        var previous = pointer[i - 1]
        pointer[i] = (
            (pointer[i] ^ (
                (previous ^ (previous >> UInt32(30))) * UInt32(1566083941)
            ))
            - UInt32(i)
        )
        i += 1
        if i >= 624:
            pointer[0] = pointer[623]
            i = 1
        count -= 1
    pointer[0] = UInt32(0x80000000)
    mt_twist(pointer)


def mt_word(state: UnsafePointer[UInt32, _], index: Int) -> UInt32:
    var word = state[index]
    word ^= word >> UInt32(11)
    word ^= (word << UInt32(7)) & UInt32(0x9D2C5680)
    word ^= (word << UInt32(15)) & UInt32(0xEFC60000)
    word ^= word >> UInt32(18)
    return word


def mt_random(state: UnsafePointer[UInt32, _], index: Int) -> Float64:
    var first = mt_word(state, index) >> UInt32(5)
    var second = mt_word(state, index + 1) >> UInt32(6)
    return (
        Float64(first) * 67108864.0 + Float64(second)
    ) * (1.0 / 9007199254740992.0)


@always_inline
def fade(value: Float64) -> Float64:
    var square = value * value
    var cube = square * value
    return cube * (value * (value * 6.0 - 15.0) + 10.0)


def perlin_sample[coord_origin: MutOrigin, tile_origin: MutOrigin](
    coordinates: UnsafePointer[Float64, coord_origin],
    dimensions: Int,
    seed: UInt64,
    tile_periods: UnsafePointer[Float64, tile_origin],
    tiled: Bool,
) -> Float64:
    var total = 0.0
    var corners = 1 << dimensions
    var hashed_lattices = Array[Float64, 8](uninitialized=True)
    var corner_weights = Array[Float64, 8](uninitialized=True)
    var corner_distances = Array[Float64, 8](uninitialized=True)
    var hashed_pointer = hashed_lattices.unsafe_ptr()
    var weight_pointer = corner_weights.unsafe_ptr()
    var distance_pointer = corner_distances.unsafe_ptr()
    comptime W = simd_width_of[DType.float64]()
    var axis = 0
    while axis + W <= dimensions:
        var coordinate_vector = coordinates.load[width=W](axis)
        var base_vector = floor(coordinate_vector)
        var upper_vector = base_vector + 1.0
        var hashed_base = base_vector
        var hashed_upper = upper_vector
        if tiled:
            var period_vector = tile_periods.load[width=W](axis)
            hashed_base -= floor(hashed_base / period_vector) * period_vector
            hashed_upper -= floor(hashed_upper / period_vector) * period_vector
        var distance_base = coordinate_vector - base_vector
        var distance_upper = distance_base - 1.0
        var fade_base = 1.0 - abs(distance_base)
        var fade_upper = 1.0 - abs(distance_upper)
        var base_square = fade_base * fade_base
        var upper_square = fade_upper * fade_upper
        hashed_pointer.store(axis, hashed_base)
        hashed_pointer.store(4 + axis, hashed_upper)
        distance_pointer.store(axis, distance_base)
        distance_pointer.store(4 + axis, distance_upper)
        weight_pointer.store(
            axis,
            base_square * fade_base * (
                fade_base * (fade_base * 6.0 - 15.0) + 10.0
            ),
        )
        weight_pointer.store(
            4 + axis,
            upper_square * fade_upper * (
                fade_upper * (fade_upper * 6.0 - 15.0) + 10.0
            ),
        )
        axis += W
    while axis < dimensions:
        var base = Int(floor(coordinates[axis]))
        for bit in range(2):
            var index = axis + bit * 4
            var lattice = base + bit
            var hashed_coordinate = Float64(lattice)
            if tiled:
                var period = tile_periods[axis]
                hashed_coordinate -= floor(hashed_coordinate / period) * period
            var distance = coordinates[axis] - Float64(lattice)
            hashed_lattices[index] = hashed_coordinate
            corner_distances[index] = distance
            corner_weights[index] = fade(1.0 - abs(distance))
        axis += 1
    for corner in range(corners):
        var hash_value = 1.0
        var place = 1.0
        var weight = 1.0
        for axis in range(dimensions):
            var bit = (corner >> (dimensions - axis - 1)) & 1
            var index = axis + bit * 4
            hash_value += place * hashed_lattices[index]
            place *= 10.0
            weight *= corner_weights[index]

        var integer_hash = Int(abs(hash_value))
        if integer_hash < 1:
            integer_hash = 1
        var state = Array[UInt32, 624](uninitialized=True)
        var combined_seed = UInt128(seed) * UInt128(integer_hash)
        if combined_seed <= UInt128(0xFFFFFFFF):
            mt_seed_one_word(state, UInt32(combined_seed))
        else:
            mt_seed(state, combined_seed)
        var pointer = state.unsafe_ptr()
        var dot_product = 0.0
        for axis in range(dimensions):
            var bit = (corner >> (dimensions - axis - 1)) & 1
            var gradient = -1.0 + 2.0 * mt_random(pointer, axis * 2)
            dot_product += gradient * corner_distances[axis + bit * 4]
        total += weight * dot_product
    return total


def perlin_batch_range(
    coordinates: FPtr,
    results: FPtr,
    start: Int,
    stop: Int,
    dimensions: Int,
    seed: UInt64,
    tile_periods: FPtr,
    tiled: Bool,
):
    for row in range(start, stop):
        results[row] = perlin_sample(
            coordinates + row * dimensions,
            dimensions,
            seed,
            tile_periods,
            tiled,
        )


@export("mpn_perlin_scalar")
def mpn_perlin_scalar(
    dimensions: Int,
    seed_value: Int,
    tiled_value: Int,
    x0: Float64,
    x1: Float64,
    x2: Float64,
    x3: Float64,
    t0: Float64,
    t1: Float64,
    t2: Float64,
    t3: Float64,
) abi("C") -> Float64:
    var coordinates = Array[Float64, 4](fill=0.0)
    var tile_periods = Array[Float64, 4](fill=1.0)
    coordinates[0] = x0
    coordinates[1] = x1
    coordinates[2] = x2
    coordinates[3] = x3
    tile_periods[0] = t0
    tile_periods[1] = t1
    tile_periods[2] = t2
    tile_periods[3] = t3
    return perlin_sample(
        coordinates.unsafe_ptr(),
        dimensions,
        UInt64(seed_value),
        tile_periods.unsafe_ptr(),
        tiled_value != 0,
    )


@export("mpn_perlin_batch")
def mpn_perlin_batch(
    coordinates_addr: Int,
    count: Int,
    dimensions: Int,
    seed_value: Int,
    tile_periods_addr: Int,
    tiled_value: Int,
    result_addr: Int,
) abi("C") -> Int:
    if dimensions < 1 or dimensions > 4 or count < 0:
        return 1
    if count == 0:
        return 0
    if coordinates_addr == 0 or result_addr == 0:
        return 1
    if tiled_value != 0 and tile_periods_addr == 0:
        return 1
    var coordinates = FPtr(unsafe_from_address=coordinates_addr)
    var results = FPtr(unsafe_from_address=result_addr)
    var tile_periods = coordinates
    if tiled_value != 0:
        tile_periods = FPtr(unsafe_from_address=tile_periods_addr)
    var tiled = tiled_value != 0
    var seed = UInt64(seed_value)
    if count < PERLIN_PARALLEL_THRESHOLD:
        perlin_batch_range(
            coordinates, results, 0, count, dimensions, seed, tile_periods, tiled
        )
        return 0
    var tasks = (count + PERLIN_PARALLEL_GRAIN - 1) // PERLIN_PARALLEL_GRAIN

    @always_inline
    def work(task: Int) {
        imm coordinates, imm results, imm count, imm dimensions,
        imm seed, imm tile_periods, imm tiled,
    }:
        var start = task * PERLIN_PARALLEL_GRAIN
        perlin_batch_range(
            coordinates,
            results,
            start,
            min(start + PERLIN_PARALLEL_GRAIN, count),
            dimensions,
            seed,
            tile_periods,
            tiled,
        )

    parallelize(
        work, tasks, min(tasks, PERLIN_PARALLEL_WORKERS)
    )
    return 0


def perlin_gpu_kernel(
    coordinates: FPtr,
    count: Int64,
    dimensions: Int64,
    seed: UInt64,
    tile_periods: FPtr,
    tiled: Int64,
    results: FPtr,
):
    var row = global_idx.x
    var dimension_count = Int(dimensions)
    if row < Int(count):
        results[row] = perlin_sample(
            coordinates + row * dimension_count,
            dimension_count,
            seed,
            tile_periods,
            tiled != 0,
        )


@export("mpn_perlin_gpu_batch")
def mpn_perlin_gpu_batch(
    coordinates_addr: Int,
    count: Int,
    dimensions: Int,
    seed_value: Int,
    tile_periods_addr: Int,
    tiled_value: Int,
    result_addr: Int,
) abi("C") -> Int:
    if dimensions < 1 or dimensions > 4 or count < 0:
        return 1
    if count == 0:
        return 0
    if coordinates_addr == 0 or tile_periods_addr == 0 or result_addr == 0:
        return 1
    var bytes_required = (count * dimensions + count + dimensions) * 8
    if bytes_required >= 2_000_000_000:
        return 2
    try:
        var coordinates = FPtr(unsafe_from_address=coordinates_addr)
        var tile_periods = FPtr(unsafe_from_address=tile_periods_addr)
        var results = FPtr(unsafe_from_address=result_addr)
        with DeviceContext() as ctx:
            var device_coordinates = (
                ctx.enqueue_create_buffer[DType.float64](count * dimensions)
            )
            var device_periods = (
                ctx.enqueue_create_buffer[DType.float64](dimensions)
            )
            var device_results = ctx.enqueue_create_buffer[DType.float64](count)
            ctx.enqueue_copy(device_coordinates, coordinates)
            ctx.enqueue_copy(device_periods, tile_periods)
            ctx.enqueue_function[perlin_gpu_kernel](
                device_coordinates,
                Int64(count),
                Int64(dimensions),
                UInt64(seed_value),
                device_periods,
                Int64(tiled_value),
                device_results,
                grid_dim=(count + 127) // 128,
                block_dim=128,
            )
            ctx.enqueue_copy(results, device_results)
            ctx.synchronize()
        return 0
    except:
        return 2


@always_inline
def grad2(index: Int, x: Float64, y: Float64) -> Float64:
    var gradient = index % 12
    if gradient == 0:
        return x + y
    if gradient == 1:
        return -x + y
    if gradient == 2:
        return x - y
    if gradient == 3:
        return -x - y
    if gradient == 4 or gradient == 6:
        return x
    if gradient == 5 or gradient == 7:
        return -x
    if gradient == 8 or gradient == 10:
        return y
    return -y


@always_inline
def grad3(index: Int, x: Float64, y: Float64, z: Float64) -> Float64:
    var gradient = index % 12
    if gradient == 0:
        return x + y
    if gradient == 1:
        return -x + y
    if gradient == 2:
        return x - y
    if gradient == 3:
        return -x - y
    if gradient == 4:
        return x + z
    if gradient == 5:
        return -x + z
    if gradient == 6:
        return x - z
    if gradient == 7:
        return -x - z
    if gradient == 8:
        return y + z
    if gradient == 9:
        return -y + z
    if gradient == 10:
        return y - z
    return -y - z


@always_inline
def perm_at(permutation: IPtr, index: Int) -> Int:
    return Int(permutation[index & 255])


def simplex2(x: Float64, y: Float64, permutation: IPtr) -> Float64:
    comptime f2 = 0.3660254037844386
    comptime g2 = 0.21132486540518713
    var skew = (x + y) * f2
    var i = Int(floor(x + skew))
    var j = Int(floor(y + skew))
    var unskew = Float64(i + j) * g2
    var x0 = x - (Float64(i) - unskew)
    var y0 = y - (Float64(j) - unskew)
    var i1 = 1 if x0 > y0 else 0
    var j1 = 0 if x0 > y0 else 1
    var x1 = x0 - Float64(i1) + g2
    var y1 = y0 - Float64(j1) + g2
    var x2 = x0 - 1.0 + 2.0 * g2
    var y2 = y0 - 1.0 + 2.0 * g2
    var ii = i & 255
    var jj = j & 255
    var gi0 = perm_at(permutation, ii + perm_at(permutation, jj))
    var gi1 = perm_at(
        permutation, ii + i1 + perm_at(permutation, jj + j1)
    )
    var gi2 = perm_at(
        permutation, ii + 1 + perm_at(permutation, jj + 1)
    )
    var value = 0.0
    var t0 = 0.5 - x0 * x0 - y0 * y0
    if t0 > 0.0:
        t0 *= t0
        value += t0 * t0 * grad2(gi0, x0, y0)
    var t1 = 0.5 - x1 * x1 - y1 * y1
    if t1 > 0.0:
        t1 *= t1
        value += t1 * t1 * grad2(gi1, x1, y1)
    var t2 = 0.5 - x2 * x2 - y2 * y2
    if t2 > 0.0:
        t2 *= t2
        value += t2 * t2 * grad2(gi2, x2, y2)
    return 70.0 * value


def simplex3(x: Float64, y: Float64, z: Float64, permutation: IPtr) -> Float64:
    comptime f3 = 1.0 / 3.0
    comptime g3 = 1.0 / 6.0
    var skew = (x + y + z) * f3
    var i = Int(floor(x + skew))
    var j = Int(floor(y + skew))
    var k = Int(floor(z + skew))
    var unskew = Float64(i + j + k) * g3
    var x0 = x - (Float64(i) - unskew)
    var y0 = y - (Float64(j) - unskew)
    var z0 = z - (Float64(k) - unskew)
    var i1 = 0
    var j1 = 0
    var k1 = 0
    var i2 = 0
    var j2 = 0
    var k2 = 0
    if x0 >= y0:
        if y0 >= z0:
            i1 = 1
            i2 = 1
            j2 = 1
        elif x0 >= z0:
            i1 = 1
            i2 = 1
            k2 = 1
        else:
            k1 = 1
            i2 = 1
            k2 = 1
    else:
        if y0 < z0:
            k1 = 1
            j2 = 1
            k2 = 1
        elif x0 < z0:
            j1 = 1
            j2 = 1
            k2 = 1
        else:
            j1 = 1
            i2 = 1
            j2 = 1
    var x1 = x0 - Float64(i1) + g3
    var y1 = y0 - Float64(j1) + g3
    var z1 = z0 - Float64(k1) + g3
    var x2 = x0 - Float64(i2) + 2.0 * g3
    var y2 = y0 - Float64(j2) + 2.0 * g3
    var z2 = z0 - Float64(k2) + 2.0 * g3
    var x3 = x0 - 1.0 + 3.0 * g3
    var y3 = y0 - 1.0 + 3.0 * g3
    var z3 = z0 - 1.0 + 3.0 * g3
    var ii = i & 255
    var jj = j & 255
    var kk = k & 255
    var gi0 = perm_at(
        permutation,
        ii + perm_at(permutation, jj + perm_at(permutation, kk)),
    )
    var gi1 = perm_at(
        permutation,
        ii + i1 + perm_at(
            permutation, jj + j1 + perm_at(permutation, kk + k1)
        ),
    )
    var gi2 = perm_at(
        permutation,
        ii + i2 + perm_at(
            permutation, jj + j2 + perm_at(permutation, kk + k2)
        ),
    )
    var gi3 = perm_at(
        permutation,
        ii + 1 + perm_at(
            permutation, jj + 1 + perm_at(permutation, kk + 1)
        ),
    )
    var value = 0.0
    var t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0
    if t0 > 0.0:
        t0 *= t0
        value += t0 * t0 * grad3(gi0, x0, y0, z0)
    var t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1
    if t1 > 0.0:
        t1 *= t1
        value += t1 * t1 * grad3(gi1, x1, y1, z1)
    var t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2
    if t2 > 0.0:
        t2 *= t2
        value += t2 * t2 * grad3(gi2, x2, y2, z2)
    var t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3
    if t3 > 0.0:
        t3 *= t3
        value += t3 * t3 * grad3(gi3, x3, y3, z3)
    return 32.0 * value


@export("mpn_simplex2_batch")
def mpn_simplex2_batch(
    coordinates_addr: Int,
    count: Int,
    permutation_addr: Int,
    octaves: Int,
    persistence: Float64,
    lacunarity: Float64,
    result_addr: Int,
) abi("C") -> Int:
    if count < 0 or octaves < 1:
        return 1
    if count == 0:
        return 0
    if coordinates_addr == 0 or permutation_addr == 0 or result_addr == 0:
        return 1
    var coordinates = FPtr(unsafe_from_address=coordinates_addr)
    var permutation = IPtr(unsafe_from_address=permutation_addr)
    var results = FPtr(unsafe_from_address=result_addr)
    for row in range(count):
        var frequency = 1.0
        var amplitude = 1.0
        var normalizer = 0.0
        var value = 0.0
        for _ in range(octaves):
            value += amplitude * simplex2(
                coordinates[row * 2] * frequency,
                coordinates[row * 2 + 1] * frequency,
                permutation,
            )
            normalizer += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        results[row] = value / normalizer
    return 0


@export("mpn_simplex3_batch")
def mpn_simplex3_batch(
    coordinates_addr: Int,
    count: Int,
    permutation_addr: Int,
    octaves: Int,
    persistence: Float64,
    lacunarity: Float64,
    result_addr: Int,
) abi("C") -> Int:
    if count < 0 or octaves < 1:
        return 1
    if count == 0:
        return 0
    if coordinates_addr == 0 or permutation_addr == 0 or result_addr == 0:
        return 1
    var coordinates = FPtr(unsafe_from_address=coordinates_addr)
    var permutation = IPtr(unsafe_from_address=permutation_addr)
    var results = FPtr(unsafe_from_address=result_addr)
    for row in range(count):
        var frequency = 1.0
        var amplitude = 1.0
        var normalizer = 0.0
        var value = 0.0
        for _ in range(octaves):
            value += amplitude * simplex3(
                coordinates[row * 3] * frequency,
                coordinates[row * 3 + 1] * frequency,
                coordinates[row * 3 + 2] * frequency,
                permutation,
            )
            normalizer += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        results[row] = value / normalizer
    return 0
