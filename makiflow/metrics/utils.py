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


def one_hot(sparse_labels, depth):
    """
    Creates one-hot encoding for sparse labels. Note that
    this method supports only 1-dimensional labels (i.e. labels grouped in batches
    or in any other order will cause unpredictable result, you have to flatten your
    labels first)
    Parameters
    ----------
    sparse_labels : array like
        Labels to encode.
    depth : int
        Num classes.
    Returns
    -------
    np.ndarray
        One-hot encoded labels.
    """
    N = len(sparse_labels)
    one_hotted = np.zeros((N, depth), dtype=np.uint8)
    one_hotted[np.arange(N), sparse_labels] = 1
    return one_hotted

