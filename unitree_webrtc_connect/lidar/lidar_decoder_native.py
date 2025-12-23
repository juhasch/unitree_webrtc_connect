# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import lz4.block

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def decompress(compressed_data, decomp_size):
    decompressed = lz4.block.decompress(
        compressed_data,
        uncompressed_size=decomp_size
    )
    return decompressed

@jit(nopython=True, cache=True)
def _bits_to_points_numba(buf_array):
    max_points = len(buf_array) * 8
    points_x = np.empty(max_points, dtype=np.int32)
    points_y = np.empty(max_points, dtype=np.int32)
    points_z = np.empty(max_points, dtype=np.int32)

    point_count = 0

    for n in range(len(buf_array)):
        byte_value = buf_array[n]
        if byte_value == 0:
            continue

        z = n // 0x800
        n_slice = n % 0x800
        y = n_slice // 0x10
        x_base = (n_slice % 0x10) * 8

        for bit_pos in range(8):
            if byte_value & (1 << (7 - bit_pos)):
                x = x_base + bit_pos
                points_x[point_count] = x
                points_y[point_count] = y
                points_z[point_count] = z
                point_count += 1

    if point_count > 0:
        return np.column_stack((points_x[:point_count], points_y[:point_count], points_z[:point_count]))
    else:
        return np.empty((0, 3), dtype=np.int32)

def bits_to_points(buf, origin, resolution=0.05):
    buf = np.frombuffer(bytearray(buf), dtype=np.uint8)

    if NUMBA_AVAILABLE:
        points = _bits_to_points_numba(buf)
    else:
        nonzero_indices = np.nonzero(buf)[0]
        points_list = []

        for n in nonzero_indices:
            byte_value = buf[n]
            z = n // 0x800
            n_slice = n % 0x800
            y = n_slice // 0x10
            x_base = (n_slice % 0x10) * 8

            for bit_pos in range(8):
                if byte_value & (1 << (7 - bit_pos)):
                    x = x_base + bit_pos
                    points_list.append((x, y, z))

        points = np.array(points_list) if points_list else np.empty((0, 3), dtype=np.int32)

    return points * resolution + origin

class LidarDecoder:
    def decode(self, compressed_data, data):
        def points():
            decompressed = decompress(compressed_data, data["src_size"])
            points = bits_to_points(decompressed, data["origin"], data["resolution"])
            return points

        return {
                "points": points(),
                # "raw": compressed_data,
        }
