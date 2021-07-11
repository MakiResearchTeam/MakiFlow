def find_key(dct, value):
    """
    Find key for the value `value` in the dictionary `dct`.

    Parameters
    ----------
    dct : dict
    value : object

    Returns
    -------
    object
        Key for the corresponding value. If the value is not present in the dct, None is returned.
    """
    for k, v in dct.items():
        if v == value:
            return k
    return None


def filter_tensors(dct1, dct2):
    """
    Merges two dictionaries containing TF tensors. If the same tensor is encountered but with different keys,
    the keys are being merged into one preserving the tensor.

    Parameters
    ----------
    dct1 : dict
    dct2 : dict

    Returns
    -------
    dict
        Merging result.
    """
    dct1, dct2 = dct1.copy(), dct2.copy()
    for tensor_name, tensor in dct2.items():
        key = find_key(dct1, tensor)

        if key is not None:
            # Remove existing pair (k, v)
            dct1.pop(key)
            tensor_name = key + '/' + tensor_name

        dct1.update({tensor_name: tensor})

    return dct1
