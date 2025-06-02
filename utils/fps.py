from utils.KDTree import KDTree, Point
import numpy as np

from scipy.spatial import distance
import tqdm

def dist(bucket, samples):
    """ Compute the far_point in a bucket given new samples/points to compare to """
    distances = distance.cdist(samples, bucket.data, "euclidean")
    min_dist = np.min(distances, axis = 0) # minimum distance of point in bucket to any point in samples

    # update bucket.distances to be the minimum distance of each point to any of the samples
    if bucket.distances is None:
        bucket.distances = min_dist
    else:
        bucket.distances = np.min([bucket.distances, min_dist], axis = 0)

    # find the new far_point and update the bucket accordingly
    minimum = np.argmax(bucket.distances, axis = 0)
    bucket.far_point = bucket.points[minimum]
    bucket.dist_far_point = bucket.distances[minimum]

def farthest_point_init(buckets, samples: [int]):
    """ Initialize the far_point of every bucket in buckets given initial sample(s) """
    f = np.vectorize(lambda b: dist(b, samples))
    f(buckets)

def satisfy_implicit(bucket, sample):
    """ Check whether the far_point of bucket is smaller than the distance
        of the new sample to the bucket """
    maxs = np.max(bucket.data, axis = 0)
    mins = np.min(bucket.data, axis = 0)

    dist = np.where(sample > maxs, sample - maxs, 0)
    dist += np.where(sample < mins, sample - mins, 0)
    distBucket = np.sqrt( np.dot(dist, dist) )

    return bucket.dist_far_point < distBucket

def satisfy_merged(bucket, sample):
    """ Check whether the distance to the far_point is smaller than the distance
        between the new sample and the current buckets far_point """
    return bucket.dist_far_point < distance.euclidean(sample, bucket.far_point.data)

def update(b, s):
    """ Update the far_point of bucket b if necessary --> based on QuickFPS """
    if satisfy_implicit(b, s):
        return b.dist_far_point
    if satisfy_merged(b, s):
        b.buffer.append(s)
        return b.dist_far_point
    # not optimized bucket
    b.buffer.append(s)
    dist(b, b.buffer)
    b.buffer = []
    return b.dist_far_point

def sample(data, n, start_indexes: [int] = None):
    """ Sample n data points from data using furthest point sampling.

    :param data: a list of multidimensional data points from which the sample is drawn.
    :param n: the size of the expected sample.
    :param start_indexes: optional, a list containing indices of an initial sample which
                          should be extended until n entries are reached.

    :return: a list of n indices pointing to the sampled points from the provided data list.
    """
    points = list(map(lambda x: Point(x[0], x[1]), list(zip(data, list(range(len(data)))))))
    tree = KDTree(data, points)
    tree.recursive_split()
    bucket = tree.buckets()

    indices = [np.random.randint(0, len(data))]
    if start_indexes is not None:
        indices = start_indexes
        if type(indices) == np.ndarray:  # it's a numpy array right now
            indices = indices.tolist()
        indices.sort()

    samples = data[indices, :]
    if type(samples) == np.ndarray:
        samples = samples.tolist()
    farthest_point_init(bucket, samples)

    for _ in tqdm.tqdm(range(len(samples), int(n))): #while len(samples) < N:
        s = samples[len(samples) - 1]
        f = np.vectorize(lambda b: update(b, s))
        i = np.argmax(f(bucket))
        far_point = bucket[i].far_point

        samples.append(far_point.data)
        indices.append(far_point.index)

    return indices
