import numpy as np


def hcv_to_num(bin_vec):
    num = 0
    for i in range(len(bin_vec)):
        num += 2**i * bin_vec[i]
    return num


def to_hc_vec(num_classes, classes):
    vec = np.zeros(num_classes)
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

