import numpy as np
import copy


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
        raise TypeError('mapping should be list of typles')

    new_masks = copy.deepcopy(masks)

    for i in range(len(new_masks)):
        for elem in mapping:
            old_value = elem[0]
            new_value = elem[1]
            new_masks[i][masks[i] == old_value] = new_value

    return  new_masks


