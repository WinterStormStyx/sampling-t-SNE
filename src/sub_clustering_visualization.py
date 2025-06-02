import numpy as np
from matplotlib import pyplot as plt

from utils import load_dataset
from utils.utils import DataType

seed = 42
NUM_THREADS = 12
perplexity_ratio = 1/7000
sampling_rate = 0.1

if __name__ == "__main__":
    # Load data
    original_data, _ = load_dataset.get_mnist(DataType.MNIST.value[0])

    # this exists if the comparison pipeline was run for perplexity = 144/7000, rate = 0.1
    samples = np.load("src/samples/sample_ids_poisson_disk_sub_clustering.npy")
    sample_labels = np.load("src/samples/labels_poisson_disk_sub_clustering.npy")
    embedding = np.load("src/samples/embedding_poisson_disk_sub_clustering.npy")

    _, axs = plt.subplots()
    plt.xticks([])
    plt.yticks([])
    axs.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=sample_labels, cmap="tab10", s=3
    )
    plt.show()

    # visualization for cluster 1 (orange)
    label_1 = embedding[sample_labels == 1, :]
    indexes_1 = samples[sample_labels == 1]

    # slanted right, severely
    group_1 = np.array([i for i in label_1 if -31 < i[0] < -24 and 20 < i[1] < 30])
    group_1_ind = [i for ind, i in enumerate(indexes_1) if -31 < label_1[ind][0] < -24 and 20 < label_1[ind][1] < 30]

    # upright
    group_1_2 = np.array([i for i in label_1 if -33 < i[0] < -22 and -18 < i[1] < -6])
    group_1_ind_2 = [i for ind, i in enumerate(indexes_1) if -33 < label_1[ind][0] < -22 and -18 < label_1[ind][1] < -6]

    # slanted right, but only slightly
    group_1_3 = np.array([i for i in label_1 if -30 < i[0] < -25 and 3 < i[1] < 10])
    group_1_ind_3 = [i for ind, i in enumerate(indexes_1) if -30 < label_1[ind][0] < -25 and 3 < label_1[ind][1] < 10]

    # visualization for cluster 6 (pink)
    label_6 = embedding[sample_labels == 6, :]
    indexes_6 = samples[sample_labels == 6]

    # line to the right, circular
    group_6_1 = np.array([i for i in label_6 if 8 < i[0] < 23 and 9 < i[1] < 23])
    group_6_ind_1 = [i for ind, i in enumerate(indexes_6) if 8 < label_6[ind][0] < 23 and 9 < label_6[ind][1] < 23]

    # more upright, less big circle
    group_6_2 = np.array([i for i in label_6 if 16 < i[0] < 20 and -6 < i[1] < 0])
    group_6_ind_2 = [i for ind, i in enumerate(indexes_6) if 16 < label_6[ind][0] < 20 and -6 < label_6[ind][1] < 0]

    # upright line and big circle which is leaning left
    group_6_3 = np.array([i for i in label_6 if 22 < i[0] < 31 and -15 < i[1] < -7.5])
    group_6_ind_3 = [i for ind, i in enumerate(indexes_6) if 22 < label_6[ind][0] < 31 and -15 < label_6[ind][1] < -7.5]

    # visualization for cluster 0 (dark blue)
    label_0 = embedding[sample_labels == 0, :]
    indexes_0 = samples[sample_labels == 0]

    # slanted to the right, severely, thin
    group_0_1 = np.array([i for i in label_0 if 27.5 < i[0] < 33 and 19 < i[1] < 25])
    group_0_ind_1 = [i for ind, i in enumerate(indexes_0) if 27.5 < label_0[ind][0] < 33 and 19 < label_0[ind][1] < 25]

    # upright, thin
    group_0_2 = np.array([i for i in label_0 if 19 < i[0] < 25 and 3 < i[1] < 7.7])
    group_0_ind_2 = [i for ind, i in enumerate(indexes_0) if 19 < label_0[ind][0] < 25 and 3 < label_0[ind][1] < 7.7]

    # slanted right, thick
    group_0_3 = np.array([i for i in label_0 if 24 < i[0] < 28 and 10 < i[1] < 16.5])
    group_0_ind_3 = [i for ind, i in enumerate(indexes_0) if 24 < label_0[ind][0] < 28 and 10 < label_0[ind][1] < 16.5]

    # really round, thin
    group_0_4 = np.array([i for i in label_0 if 38 < i[0] < 45 and -1 < i[1] < 7])
    group_0_ind_4 = [i for ind, i in enumerate(indexes_0) if 38 < label_0[ind][0] < 45 and -1 < label_0[ind][1] < 7]

    # for a specific group, actually show the data points in it
    for ind in group_1_ind:
        first_image = original_data[ind].reshape((28, 28))
        plt.imshow(first_image, cmap="grey")
        plt.show()
