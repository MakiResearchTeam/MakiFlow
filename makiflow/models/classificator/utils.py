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
import copy

import matplotlib.pyplot as plt
import numpy as np


plt.switch_backend('agg')
# For loading dendrite images


def error_rate(Y, T):
    return np.mean(Y != T)


def cross_entropy(Y, T):
    return -np.mean(T * np.log(Y))


def sparse_cross_entropy(Y, T):
    return -np.mean(np.log(Y[np.arange(Y.shape[0]), T]))


def mutate_masks(masks, mapping):
    """
    Remaps classes on the given `masks` according to the `mapping`.

    Parameters
    ----------
    masks : list or numpy.array
        List or numpy array of masks.
    mapping : list
        List of tuples: [(source_class_number, new_class_number)],
        where `source_class_number` will be changed to `new_class_number` in the `masks`.

    Returns
    ---------
    new_masks : the same type as `masks`
        New masks with changed class numbers.
    """
    if type(mapping) is not list or (len(mapping) != 0 and type(mapping[0]) is not tuple):
        raise TypeError('mapping should be list of tuples')

    new_masks = copy.deepcopy(masks)

    for i in range(len(new_masks)):
        for elem in mapping:
            old_value = elem[0]
            new_value = elem[1]
            new_masks[i][masks[i] == old_value] = new_value

    return new_masks


def merging_masks(masks, index_of_main_mask, priority):
    """
    We choose the core mask which have index `index_of_main_mask` on which is put other classes
    according to `priority` of other classes

    Parameters
    ----------
    masks : list or numpy.array
        List or numpy array of masks.
    index_of_main_mask : int
        Index of core mask.
    priority : list
        List of priority of classes. Classes will be installed in the prescribed manner,
        from highest priority to least priority in the list (from left to right in list).
        Example: [9, 10, 3, 7], where 9th class have highest priority and will be put on others classes the first,
        on the other hand the 7th class have the lowest priority and will be placed last in the mask,
        but not including those that have already been replaced.
    Returns
    ---------
    main_mask : numpy.array
        The core mask after merging other classes.
    """

    # For easy access and stay `masks[index_of_main_mask]` unchanged
    main_mask = copy.deepcopy(masks[index_of_main_mask])

    # Thanks to the `boolean_mask` higher priority class will not be erased
    boolean_mask = np.ones(main_mask.shape).astype(np.bool)

    for i in range(len(masks)):
        if i == index_of_main_mask:
            continue

        for prior in priority:
            # Take indexes where class `prior` is in `masks[i]`
            temp_bool_mask = masks[i] == prior
            # Remove indexes that have already been used
            temp_bool_mask = temp_bool_mask * boolean_mask

            main_mask[temp_bool_mask] = prior
            boolean_mask[masks[i] == prior] = False

    return main_mask
