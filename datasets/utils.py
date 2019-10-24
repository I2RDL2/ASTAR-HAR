# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np


def random_partitions(data, first_size, random):
    """Split data into two random partitions of sizes n and len(data) - n

    Args:
        data (ndarray): data to be split
        first_size (int): size of the first partition
        random (RandomState): source of randomness

    Return:
        tuple of two ndarrays
    """
    mask = np.zeros(len(data), dtype=bool)
    mask[:first_size] = True
    random.shuffle(mask)
    return data[mask], data[~mask]


def random_balanced_partitions(data, first_size, labels, random=np.random):
    """Split data into a balanced random partition and the rest

    Partition the `data` array into two random partitions, using
    the `labels` array (of equal size) to guide the choice of
    elements of the first returned array.

    Example:
        random_balanced_partition(['a', 'b', 'c'], 2, [3, 5, 5])
        # Both labels 3 and 5 need to be presented once, so
        # the result can be either (['a', 'b'], ['c']) or
        # (['a', 'c'], ['b']) but not (['b', 'c'], ['a']).

    Args:
        data (ndarray): data to be split
        first_size (int): size of the first partition
        balance (ndarray): according to which balancing is done
        random (RandomState): source of randomness

    Return:
        tuple of two ndarrays
    """
    assert len(data) == len(labels)

    classes, class_counts = np.unique(labels, return_counts=True)
    assert len(classes) <= 10000, "surprisingly many classes: {}".format(len(classes))
    # assert first_size % len(classes) == 0, "not divisible: {}/{}".format(first_size, len(classes))
    # fix problem when first_size is not divisible by classes
    if first_size % len(classes)!=0:
        print("not divisible: {}/{}".format(first_size, len(classes)))
        first_size = first_size- first_size%len(classes)

    # assert np.all(class_counts >= first_size // len(classes)), "not enough examples of some class"
    # original code may bring warning of "not enough examples of some classes" because of inbalanced samples
    # for example collecting 1000 samples from [1000,400] datasets, supposed to get [500, 500]
    idxs_per_class = [np.nonzero(labels == klass)[0] for klass in classes]
    chosen_idxs_per_class = [
        # random.choice(idxs, int(first_size / sum(class_counts) * len(idxs)), replace=False)
        random.choice(idxs, int(first_size/len(classes)), replace=False)
        for idxs in idxs_per_class
    ]

    first_idxs = np.concatenate(chosen_idxs_per_class)
    second_idxs = np.setdiff1d(np.arange(len(labels)), first_idxs)

    assert len(first_idxs) + len(second_idxs) == len(data), "Not all data are used for training!"
    return data[first_idxs], data[second_idxs]


def random_ratio_partitions(data, first_size, labels, random=np.random):

    assert len(data) == len(labels)

    classes, class_counts = np.unique(labels, return_counts=True)
    assert len(classes) <= 10000, "surprisingly many classes: {}".format(len(classes))

    if first_size % len(classes)!=0:
        print("not divisible: {}/{}".format(first_size, len(classes)))
        first_size = first_size- first_size%len(classes)

    idxs_per_class = [np.nonzero(labels == klass)[0] for klass in classes]

    most_num_classes = np.argmax(class_counts)
    most_idxs = idxs_per_class.pop(most_num_classes)

    chosen_idxs_per_class = [
        random.choice(idxs, int(first_size/len(data)*len(idxs)), replace=False)
        for idxs in idxs_per_class ]

    chosen_idxs_per_class_part2 = [
        random.choice(most_idxs, first_size - len(np.concatenate(chosen_idxs_per_class)), replace=False)
    ]
 
    chosen_idxs_per_class.extend(chosen_idxs_per_class_part2)    
    first_idxs = np.concatenate(chosen_idxs_per_class)
    second_idxs = np.setdiff1d(np.arange(len(labels)), first_idxs)

    assert len(first_idxs) + len(second_idxs) == len(data), "Not all data are used for training!"
    return data[first_idxs], data[second_idxs]