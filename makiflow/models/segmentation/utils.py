import numpy as np
import copy


def mutate_masks(masks, mapping):
    """
    Change certain class on masks on new value
    mapping is the follow list of tuples: [(source_class_number, new_class_number)]
    where source_class_number will be changed to new_class_number in the mask
    Parameters
    ----------
    masks : list or numpy.array
        List or numpy array of masks
    mapping : list
        List for changes classes number

    Returns
    ---------
    new_masks : list
        New masks with changed number of classes
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
