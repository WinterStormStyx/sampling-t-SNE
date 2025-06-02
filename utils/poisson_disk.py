import numpy as np
import tqdm

def sample(data, n, weights):
    """ Sample n data points from data using Poisson disk sampling.

    :param data:    a list of multidimensional data points from which the sample is drawn.
    :param n:       the size of the expected sample.
    :param weights: the p-matrix containing probabilities between high-dimensional neighbours.
    :return: a list of n indices pointing to the sampled points from the provided data list.
    """
    point_weights = np.array(np.sum(weights, axis=0))[0]

    removed = set()

    for _ in tqdm.tqdm(range(len(data) - int(n))):
        sample = np.argmax(point_weights)
        point_weights -= weights[sample].toarray()[0]
        point_weights[sample] = 0 # the sample has been added, it's weight is now 0
        removed.add(sample)

    return list(set(range(len(data))) - removed)

