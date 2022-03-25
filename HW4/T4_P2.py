# CS 181, Spring 2022
# Homework 4

import numpy as np
import matplotlib.pyplot as plt

import math
from numpy.random import default_rng
import copy

# Loading datasets for K-Means and HAC
small_dataset = np.load("data/small_dataset.npy")
large_dataset = np.load("data/large_dataset.npy")

# NOTE: You may need to add more helper functions to these classes
class KMeans(object):
    # K is the K in KMeans
    def __init__(self, K):
        self.K = K
        self.X = large_dataset
        self.z = np.zeros((self.X.shape[0], self.K))
        # Randomly choose points to be cluster centers
        self.initialize_cluster_centers()
        self.past_mean_images = None

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        self.past_mean_images = self.get_mean_images()
        self.z = np.zeros((self.X.shape[0], self.K))
        rss = 0
        for i, point in enumerate(X):
            distances = np.sum(np.subtract(self.past_mean_images, point) ** 2, axis=1)
            self.z[i][np.argmin(distances)] = 1
            rss += min(distances)
        return rss

    # This should return the arrays for K images. Each image should represent the mean of each of the fitted clusters.
    def get_mean_images(self):
        clusters = []
        for k in range(self.z.shape[1]):
            cluster = []
            for n in range(self.z.shape[0]):
                if self.z[n][k] == 1:
                    cluster.append(self.X[n])
            clusters.append(cluster)

        mean_images = []
        for i, cluster in enumerate(clusters):
            # Edge case of cluster being empty: according to ed post #456, continue to next iteration without changing the centroid
            if len(cluster) == 0:
                mean_images.append(self.past_mean_images[i])
            else:
                mean_images.append(np.array(np.sum(cluster, axis=0) / len(cluster)))
        return np.array(mean_images)

    def initialize_cluster_centers(self):
        # Randomly choose self.K points to use as cluster centers.
        rng = default_rng()
        points = rng.choice(self.X.shape[0], size=self.K, replace=False)
        for i, point in enumerate(points):
            self.z[point][i] = 1

    # Check if we have converged on a stable set of clusters.
    @property
    def converged(self):
        if self.past_mean_images is None:
            return False
        if np.array_equal(self.past_mean_images, self.get_mean_images()):
            return True
        return False


class HAC(object):
    def __init__(self, linkage):
        self.linkage = linkage

    # X is a (N x 784) array since the dimension of each image is 28x28.
    def fit(self, X):
        pass

    # Returns the mean image when using n_clusters clusters
    def get_mean_images(self, n_clusters):
        # TODO: Change this!
        return small_dataset[:n_clusters]


# ~~ Part 1 ~~
K = 10
kmeans = KMeans(K)
rss = []

while not kmeans.converged:
    rss.append(kmeans.fit(large_dataset))

plt.plot([i for i in range(len(rss))], rss, "o", linestyle="-")
plt.title("K-means objective function vs. iterations")
plt.xlabel("RSS")
plt.ylabel("Iterations")
plt.savefig("2-1.png")
plt.show()


# Plotting code for parts 2 and 3
def make_mean_image_plot(data, standardized=False):
    # Number of random restarts
    niters = 3
    K = 10

    # Standardize data
    if standardized:
        new_data = copy.deepcopy(data)
        for i in range(data.shape[1]):
            mean = np.mean(data[:][i])
            std = np.std(data[:][i])
            for j in range(data.shape[0]):
                new_data[j][i] -= mean
                new_data[j][i] /= std if std != 0 else 1
        data = new_data

    # Will eventually store the pixel representation of all the mean images across restarts
    allmeans = np.zeros((K, niters, 784))
    for i in range(niters):
        KMeansClassifier = KMeans(K=K)
        KMeansClassifier.fit(data)
        allmeans[:, i] = KMeansClassifier.get_mean_images()
    fig = plt.figure(figsize=(10, 10))
    plt.suptitle(
        "Class mean images across random restarts"
        + (" (standardized data)" if standardized else ""),
        fontsize=16,
    )
    for k in range(K):
        for i in range(niters):
            ax = fig.add_subplot(K, niters, 1 + niters * k + i)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis="both", which="both", length=0)
            if k == 0:
                plt.title("Iter " + str(i))
            if i == 0:
                ax.set_ylabel("Class " + str(k), rotation=90)
            plt.imshow(allmeans[k, i].reshape(28, 28), cmap="Greys_r")
    plt.show()


# ~~ Part 2 ~~
make_mean_image_plot(large_dataset, False)

# ~~ Part 3 ~~
# TODO: Change this line! standardize large_dataset and store the result in large_dataset_standardized
large_dataset_standardized = large_dataset
make_mean_image_plot(large_dataset_standardized, True)

# Plotting code for part 4
LINKAGES = ["max", "min", "centroid"]
n_clusters = 10

fig = plt.figure(figsize=(10, 10))
plt.suptitle("HAC mean images with max, min, and centroid linkages")
for l_idx, l in enumerate(LINKAGES):
    # Fit HAC
    hac = HAC(l)
    hac.fit(small_dataset)
    mean_images = hac.get_mean_images(n_clusters)
    # Make plot
    for m_idx in range(mean_images.shape[0]):
        m = mean_images[m_idx]
        ax = fig.add_subplot(
            n_clusters, len(LINKAGES), l_idx + m_idx * len(LINKAGES) + 1
        )
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
        if m_idx == 0:
            plt.title(l)
        if l_idx == 0:
            ax.set_ylabel("Class " + str(m_idx), rotation=90)
        plt.imshow(m.reshape(28, 28), cmap="Greys_r")
plt.show()

# TODO: Write plotting code for part 5

# TODO: Write plotting code for part 6
