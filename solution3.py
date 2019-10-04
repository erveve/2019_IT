import numpy as np
from math import floor, ceil

def get_indices(N, n_batches, split_ratio):
    """Generates splits of indices from 0 to N-1 into uniformly distributed\
       batches. Each batch is defined by 3 indices [i, j, k] where\
       (j-i) = split_ratio*(k-j). The first batch starts with i = 0,\
       the last one ends with k = N - 1.
    Args:
        N (int): total counts
        n_batches (int): number of splits
        split_ratio (float): split ratio, defines position of j in [i, j, k].

    Returns:
        generator for batch indices [i, j, k]
    """
    min_err = 0.5
    opt_step = 1
    for step in range(1,ceil((N-1)/n_batches)):
        tmp = (N-1 - (n_batches-1)*step)/(split_ratio+1)
        diff = (tmp - floor(tmp)) if (tmp - floor(tmp))<0.5  else (ceil(tmp) - tmp)
        if diff <= min_err:
            min_err = diff
            opt_step = step

    inds = np.array([0, 0, 0])
    for i in range(n_batches):
        if i==0:
            inds[2] = N-1 - (n_batches-1)*opt_step
            tmp = inds[2]/(split_ratio+1)
            inds[1] = tmp if (tmp - floor(tmp))<0.5 else (tmp+1)
        else:
            inds += np.array([opt_step]*3)
        # todo: move forward batch
        # calculate new indices
        yield inds

def main():
    for inds in get_indices(100, 5, 0.25):
        print(inds)
    # expected result:
    # [0, 44, 55]
    # [11, 55, 66]
    # [22, 66, 77]
    # [33, 77, 88]
    # [44, 88, 99]

if __name__ == "__main__":
    main()