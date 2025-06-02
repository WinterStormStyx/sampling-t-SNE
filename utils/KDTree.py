import numpy as np

class Point:
    """ A class to store a data point, and it's index in the overall dataset"""
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def data(self):
        return self.data()

    def __str__(self):
        return str(self.data)

class KDTree:
    """ A class to split a large dataset of high dimensionality into a KDTree """
    def __init__(self, data, points, height = 0):
        self.data = data
        self.points = points # list of points
        self.distances = None
        self.height = height
        self.left = None
        self.right = None
        self.far_point = None
        self.dist_far_point = 0
        self.buffer = []

    def split(self):
        """ Split the current level of the KDTree based on highest variance dimension """
        centers = np.var(self.data, axis = 0)
        dim = np.argmax(centers)
        center = np.average(self.data, axis = 0)[dim]

        cond = self.data[:, dim] <= center
        left, right = self.data[cond], self.data[~cond]

        l = np.extract(cond, self.points)
        r = np.extract(~cond, self.points)

        self.left = KDTree(left, l, self.height + 1)
        self.right = KDTree(right, r, self.height + 1)

        self.data = None

    def recursive_split(self):
        """ Recursively split the data in the tree until there are less than 200 datapoints
            per leaf, or a maximum height of 8 is reached """
        if (self.data is not None and len(self.data) < 200) or self.height >= 8:
            return

        if self.data is not None:
            self.split()
        self.left.recursive_split()
        self.right.recursive_split()

    def buckets(self):
        """ Find all the leaf buckets of the tree (the ones containing data) """
        if self.data is None:
            return np.concatenate((self.left.buckets(), self.right.buckets()))
        return [self]
