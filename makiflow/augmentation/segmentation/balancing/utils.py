# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def hcv_to_num(bin_vec):
    num = 0
    for i in range(len(bin_vec)):
        num += int(2**i * bin_vec[i])
    return num


def to_hc_vec(num_classes, classes):
    vec = np.zeros(num_classes, dtype=np.uint8)
    vec[classes] = 1
    return vec

def get_unique(arr):
    uniq = {}
    uniq_vecs = []
    for vec in arr:
        num = hcv_to_num(vec)
        uniq[num] = 1 + uniq.get(num, 0)
        if uniq[num] == 1:
            uniq_vecs += [vec]
    return np.array(uniq_vecs), uniq

