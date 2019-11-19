import numpy as np


def to_one_hot(indices, num_actions):
    """
    Converts a list of indices to a one-hot representation.

    For example, to_one_hot((1,3),4) -> [[0,1,0,0], [0, 0, 0, 1]]
    Args:
        indices:
        num_actions: Number of possible actions. (length of one-hot vector)
    """
    indices = np.atleast_1d(indices).astype(dtype=np.int32, copy=False)
    n_rows = len(indices)
    arr = np.zeros((n_rows, num_actions))
    arr[np.arange(n_rows), indices] = 1
    return arr
