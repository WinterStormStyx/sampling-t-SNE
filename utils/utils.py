import os
import pickle
import numpy as np
from time import time
from openTSNE import affinity, TSNEEmbedding
from openTSNE.nearest_neighbors import Annoy
from scipy.sparse import csr_matrix
from enum import Enum

def distance_matrix_and_annoy(data, num_neighbors, epsilon=False, num_threads=1, rnd_state=42):
    """ Method from Skrodzki et al. paper to compute the distance matrix and annoy of the data

    :param data:           the data that will be embedded
    :param num_neighbors:  the number of neighbors that should be maintained
    :param epsilon:        the smallest value
    :param num_threads:    the number of threads to use when computing
    :param rnd_state:      the random state to use when computing values
    :return: high_dimensional_distance_matrix and Annoy object of the data
    """
    # Create an Annoy object to get the neighborhood indices and the corresponding distances
    annoy = Annoy(
        data=data,
        k=num_neighbors,
        metric="euclidean",
        n_jobs=num_threads,
        random_state=rnd_state,
        verbose=False
    )
    neighbors, distances = annoy.build()
    # Remove explicit zero entries
    if epsilon:
        distances[distances == 0.0] = epsilon
    # Convert the information to a CSR matrix
    row_indices = np.repeat(
        np.arange(data.shape[0]),
        num_neighbors
    )
    data_distance_matrix = csr_matrix(
        (distances.flatten(), (row_indices, neighbors.flatten())),
        shape=(data.shape[0], data.shape[0])
    )
    return data_distance_matrix, annoy

def high_dimensional_distance_matrix_and_affinities(
        data_to_be_embedded, num_neighbors, perplexity, epsilon=False, num_threads=1, rnd_state=42
):
    """ Method from Skrodzki et al. paper to compute the distance matrix and affinities of the data

    :param data_to_be_embedded: the data that will be embedded
    :param num_neighbors:       the number of neighbors that should be maintained
    :param perplexity:          the perplexity value used for creating the embedding
    :param epsilon:             smallest value
    :param num_threads:         the number of threads to use when computing
    :param rnd_state:           the random state to use when computing values
    :return: computed high_dimensional_distance_matrix and the affinities of the data
    """
    high_dimensional_distance_matrix, annoy = distance_matrix_and_annoy(
        data_to_be_embedded, num_neighbors, epsilon, num_threads, rnd_state
    )
    # Compute the affinities (P matrix) to be used in t-SNE
    affinities = affinity.PerplexityBasedNN(
        knn_index=annoy,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=num_threads,
        random_state=rnd_state,
        verbose=False
    )
    return high_dimensional_distance_matrix, affinities

def load_or_compute_high_dimensional_distance_matrix_and_affinities(
        data_directory, data_name, data_to_be_embedded, num_neighbors, perplexity, epsilon=False, num_threads=1,
        rnd_state=42
):
    """ Method from Skrodzki et al. paper to load of compute the distance matrix and affinities of the data

    :param data_directory:      string of the path to the distance matrix directory - should end with `/`
    :param data_name:           string of the name of the data being loaded
    :param data_to_be_embedded: the data that will be embedded
    :param num_neighbors:       the number of neighbors that should be maintained
    :param perplexity:          the perplexity value used for creating the embedding
    :param epsilon:             smallest value
    :param num_threads:         the number of threads to use when computing
    :param rnd_state:           the random state to use when computing values
    :return: distances between neighbours as well as the affinities of the data
    """
    # Determine the high-dimensional affinities (P matrix) of the data
    if (os.path.isfile(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl")
            and os.path.isfile(
                data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl")):
        # Found the high-dimensional distance matrix and the affinities from a previous run, load those
        print("Loading high-dimensional distance matrix and affinities from saved pickle.")
        with (open(data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl", "rb")
              as high_dimensional_distance_matrix_file):
            high_dimensional_distance_matrix = pickle.load(high_dimensional_distance_matrix_file)
            high_dimensional_distance_matrix_file.close()
        with open(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl", "rb") as affinity_file:
            print(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl")
            affinities = pickle.load(affinity_file)
            affinity_file.close()
    else:
        # Could not find high-dimensional distance matrix or the affinities, compute and store them
        print("Computing affinities from scratch.")
        affinities_start = time()
        high_dimensional_distance_matrix, affinities = high_dimensional_distance_matrix_and_affinities(
            data_to_be_embedded, num_neighbors, perplexity, epsilon, num_threads, rnd_state
        )
        with (open(data_directory + data_name + f"_perplexity_{perplexity}_high_dimensional_distance_matrix.pkl", "wb")
              as high_dimensional_distance_matrix_file):
            pickle.dump(high_dimensional_distance_matrix, high_dimensional_distance_matrix_file)
            high_dimensional_distance_matrix_file.close()
        with open(data_directory + data_name + f"_perplexity_{perplexity}_affinities.pkl", "wb") as affinity_file:
            pickle.dump(affinities, affinity_file)
            affinity_file.close()
        print(f"Computing affinities took {time() - affinities_start} seconds.")

    return high_dimensional_distance_matrix, affinities


def sample_tsne(
        sampled_initial_embedding, sampled_data, sample_perplexity,
        num_iterations_early_exaggeration=250, num_iterations_optimization=750,
        n_jobs=1, rnd_state=42, callbacks=None, callbacks_every_iters=50, verbose=False
):
    """ Method inspired by Skrodzki et. al paper to compute the embedding of a sample

    :param sampled_initial_embedding:         the sample of the initial embedding of the data
    :param sampled_data:                      the actual initial data in the sample
    :param sample_perplexity:                 the perplexity value to be used on the sample
    :param num_iterations_early_exaggeration: number of iterations that should be completed as exaggeration
    :param num_iterations_optimization:       number of iterations that should be completed as optimization
    :param n_jobs:                            number of threads to use while running t-SNE
    :param rnd_state:                         the random state to be used when computing the embedding
    :param callbacks:
    :param callbacks_every_iters:
    :param verbose:                           whether the embedding should return more information
    :return: the embedding of the sample as well as the distance matrix of the data
    """
    # Compute sample embedding
    start_sample = time()

    num_neighbors = min(sampled_data.shape[0] - 1, int(3 * sample_perplexity))
    high_dimensional_distance_matrix, annoy = distance_matrix_and_annoy(
        sampled_data, num_neighbors, rnd_state= rnd_state
    )
    # Compute the affinities (P matrix) to be used in t-SNE
    sample_affinities = affinity.PerplexityBasedNN(
        knn_index=annoy,
        perplexity=sample_perplexity,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=rnd_state,
        verbose=False
    )

    sample_embedding = TSNEEmbedding(
        sampled_initial_embedding,
        sample_affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        callbacks=callbacks,
        callbacks_every_iters=callbacks_every_iters,
        verbose=verbose,
        random_state=rnd_state
    )

    # Initial optimization with exaggeration
    sample_embedding = sample_embedding.optimize(
        n_iter=num_iterations_early_exaggeration,
        exaggeration=12,
        momentum=0.5
    )

    # Further optimization without exaggeration
    sample_embedding = sample_embedding.optimize(
        n_iter=num_iterations_optimization,
        exaggeration=1,
        momentum=0.5
    )

    print(f"Sampling and sample embedding took {time() - start_sample} seconds.")


    return sample_embedding, high_dimensional_distance_matrix

def base_tsne(
        initial_embedding, affinities, negative_gradient_method="fft", num_iterations_early_exaggeration=250,
        num_iterations_optimization=750, n_jobs=1, rnd_state=42, callbacks=None, callbacks_every_iters=50, verbose=False
):
    """ Method from the Skrodzki et al. paper that perform a basic t-SNE embedding on the data.
    First perform a number of "early exaggeration" iterations, followed by a number of non-exaggerated iterations.

    :param ndarray initial_embedding:         the initial embedding of the dataset
    :param affinities:                        the affinities of the dataset
    :param str negative_gradient_method:      one of "fft" or "bh", specifying the usage of the corresponding acceleration.
    :param num_iterations_early_exaggeration: number of iterations for the early exaggeration step
    :param num_iterations_optimization:       number of iterations for the optimization step
    :param int n_jobs:                        number of threads to use for the computation
    :param int rnd_state:                     the random state that should be used for the computation
    :param int callbacks:
    :param int callbacks_every_iters:
    :param bool verbose:                      whether the embedding should return more data
    :return: The time it took to compute the embedding and a 2D embedding of the data.
    """
    start = time()

    # Set up the optimization
    t_sne_embedding = TSNEEmbedding(
        initial_embedding,
        affinities,
        negative_gradient_method=negative_gradient_method,
        n_jobs=n_jobs,
        callbacks=callbacks,
        callbacks_every_iters=callbacks_every_iters,
        verbose=verbose,
        random_state=rnd_state
    )

    # Initial optimization with exaggeration
    embedding_after_early_exaggeration = t_sne_embedding.optimize(
        n_iter=num_iterations_early_exaggeration,
        exaggeration=12,
        momentum=0.5
    )

    # Final optimization without exaggeration
    final_embedding = embedding_after_early_exaggeration.optimize(
        n_iter=num_iterations_optimization,
        exaggeration=1,
        momentum=0.5
    )

    end = time()

    return end - start, final_embedding

class SampleType(Enum):
    """ Enum specifying type of sampling method supported, with value being string representation """
    UNIFORM = "UNIFORM"
    FPS = "FPS"
    POISSON_DISK = "POISSON_DISK"
    RANDOM_WALK = "RANDOM_WALK"

class DataType(Enum):
    """ Enum specifying the type of data, with the path to the data and the label for the data as the value """
    MNIST = "data/mnist", "mnist"
    CELEGANS = "data/c_elegans_myeloid_planaria", "celegans"
    WONG = "data/wong", "wong"
