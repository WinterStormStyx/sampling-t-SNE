from numpy.f2py.auxfuncs import throw_error

from openTSNE import initialization
from matplotlib import pyplot as plt
from sklearn.metrics import auc

from utils.load_dataset import get_mnist, load_celegans, load_wong
from utils.precision_recall import precision_recall_curves
from utils import fps, poisson_disk, random_walk

from utils.utils import *


NUM_THREADS = 12
k_max = 12

def sample_with_provided_strategy(data, sample_size, sample_type: SampleType, p_matrix, perplexity, seed=42):
    """ Find a list of sampled indices from data using the provided sample_type

    :param data:        the data from which a sample should be drawn
    :param sample_size: the size of the sample that should be returned
    :param sample_type: one of the SampleType keys detailing the sampling strategy that should be used
    :param p_matrix:    the p_matrix of the high-dimensional data
    :param perplexity:  the perplexity that was used for making the p_matrix
    :param seed:        the seed that should be used for the sampling - default to 42
    :return: A list of size sample_size containing the indices of the drawn sample
    """
    np.random.seed(seed)

    if sample_type == SampleType.UNIFORM:
        sample_ids = np.random.choice(np.arange(0, data.shape[0]), size=int(sample_size), replace=False)

    elif sample_type == SampleType.FPS:
        rates = np.array([0.85, 0.7, 0.55, 0.4, 0.25, 0.1]) * len(data)
        rates = np.array([int(i) for i in rates])
        start = None
        for val in np.extract(rates <= sample_size, rates):
            try:
                start = np.load("src/samples/fps_data_{}_sample_{}_seed_{}.npy".format(len(data), val, seed))
                break
            except FileNotFoundError:
                continue
        sample_ids = fps.sample(data, sample_size, start_indexes=start)

    elif sample_type == SampleType.POISSON_DISK:
        sample_ids = poisson_disk.sample(data, sample_size, p_matrix)

    elif sample_type == SampleType.RANDOM_WALK:
        try:
            sample_ids = np.load("src/samples/random_walk_counter_{}_perplexity_{}_seed_{}.npy".format(len(data), perplexity, seed))[-int(sample_size):]
        except FileNotFoundError:
            sample_ids = random_walk.sample(data, sample_size, p_matrix, info = (perplexity, seed))

    else:
        throw_error("not matching type")
        return None

    sample_ids.sort()
    return sample_ids

def load_data(data_type: DataType, pca_dimensions = 50, seed = 42):
    """ Load the data and labels from the expected location of the dataset

    :param data_type:       value of type DataType storing the data and its path to be loaded
    :param pca_dimensions:  the number of dimensions the data should be initially reduced to
    :param seed:            the random seed that should be used in the initial reduction
    :return: the data as well as the labels for this data
    """
    if data_type == DataType.MNIST:
        data, labels = get_mnist(data_type.value[0])

    elif data_type == DataType.CELEGANS:
        data, labels = load_celegans(data_type.value[0])
        data = data.toarray()

    elif data_type == DataType.WONG:
        data, labels = load_wong(data_type.value[0])

    else:
        throw_error("can't load that data")
        return None # threw an error

    # reduce the dimensions of the dataset to 50
    data = initialization.pca(data, n_components=pca_dimensions, random_state=seed)
    return data, labels

def comparison_pipeline(data_type: DataType, perplexity_ratio, sampling_rate, seed = 42):
    """ Print the Area under the Precision Recall Curve for all sampling methods

    :param data_type:        the type of data that should be loaded
    :param perplexity_ratio: the perplexity ratio to be used
    :param sampling_rate:    the sampling rate that should be used
    :param seed:             the seed that should be used for the embedding
    """
    data_to_be_embedded, labels = load_data(data_type, seed=seed)

    initial_embedding = initialization.pca(data_to_be_embedded, random_state=seed)

    full_perplexity = perplexity_ratio * len(data_to_be_embedded)
    sample_perplexity = full_perplexity * sampling_rate
    sample_size = int(sampling_rate * len(data_to_be_embedded))
    num_neighbors = min(data_to_be_embedded.shape[0] - 1, int(3 * full_perplexity))

    p_matrix_init, affinity = load_or_compute_high_dimensional_distance_matrix_and_affinities(
        "data/", data_type.value[1], data_to_be_embedded, num_neighbors, full_perplexity,
        epsilon=False, num_threads=NUM_THREADS, rnd_state=seed
    )


    for sample_type in SampleType:
        sample_ids = sample_with_provided_strategy(data_to_be_embedded, sample_size, sample_type, p_matrix_init, perplexity_ratio, seed=seed)
        sampled_data = data_to_be_embedded[sample_ids, :]
        sampled_initial_embedding = initial_embedding[sample_ids, :]
        sampled_labels = labels[sample_ids]

        if sample_type == SampleType.FPS:
            np.save("src/samples/fps_data_{}_sample_{}_seed_{}".format(len(data_to_be_embedded), len(sampled_data), seed), sample_ids)

        sample_embedding, p_matrix = sample_tsne(
            sampled_initial_embedding, sampled_data, sample_perplexity, rnd_state=seed
        )

        if sample_type == SampleType.POISSON_DISK and perplexity_ratio == 144/7000 and sampling_rate == 0.1:
            np.save("src/samples/sample_ids_poisson_disk_sub_clustering.npy", sample_ids)
            np.save("src/samples/labels_poisson_disk_sub_clustering.npy", sampled_labels)
            np.save("src/samples/embedding_poisson_disk_sub_clustering.npy", sample_embedding)

        _, axs = plt.subplots()
        plt.xticks([])
        plt.yticks([])
        axs.scatter(
            sample_embedding[:, 0],
            sample_embedding[:, 1],
            c=sampled_labels, cmap="tab10", s=3
        )
        plt.savefig("src/figures/{}_{}_perp_{}_sample_{}_seed_{}.png".format(data_type.value[1], sample_type.value, perplexity_ratio, sampling_rate, seed))
        plt.close()

        precision, recall = precision_recall_curves(p_matrix, sample_embedding, k_max = k_max)

        print(sample_type, perplexity_ratio, sampling_rate, auc(recall, precision))
