import tqdm
import numpy as np

def random_walk(start_index, n, transitions):
    """ Find where a random walk of length n given start index and transition probabilities would end

    :param start_index: the start index of the random walk
    :param n:           how long the random walk should be
    :param transitions: the transition probabilities
    :return: the index of the end of the walk
    """
    for _ in range(n):
        choices = np.array(transitions[start_index].toarray())[0]

        choices = np.cumsum(choices)

        val = np.random.random() * choices[-1]

        start_index = len(choices[choices < val])

    return start_index

def sample(data, n, p_matrix, info = None):
    """ Sample n data points from data using Random walks.

    :param data:     the data from which a sample should be computed
    :param n:        the size of the sample
    :param p_matrix: distances between neighbouring data points
    :param info:     additional information: if None the sample will not be saved
                     if data should be saved, format this as a tuple of (perplexity, seed)
    :return: a list of n indices pointing to the sampled points from the provided data list.
    """

    num_data_points = len(data)
    counter = np.zeros(num_data_points)

    # p_matrix now holds transition probabilities
    f = np.vectorize(lambda x: np.exp(x ** 2))
    p_matrix.data = f(p_matrix.data)

    for point in tqdm.tqdm(range(num_data_points)):
        for _ in range(5):
            ind = random_walk(point, 30, p_matrix)
            counter[ind] += 1

    indexes = np.argpartition(counter, num_data_points - 1)

    if info is not None:
        perplexity, seed = info
        np.save("src/samples/random_walk_counter_{}_perplexity_{}_seed_{}".format(num_data_points, perplexity, seed), indexes)

    return indexes[-int(n):]
