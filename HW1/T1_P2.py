#####################
# CS 181, Spring 2022
# Homework 1, Problem 2
# Start Code
##################

from audioop import reverse
import math
import matplotlib.cm as cm

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

# set up data
data = [(0.0, 0.0), (1.0, 0.5), (2.0, 1), (3.0, 2), (4.0, 1), (6.0, 1.5), (8.0, 0.5)]

df = pd.DataFrame(data, columns=["x", "y"])

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])

x_test = np.arange(0, 12, 0.1)
x_test_df = pd.DataFrame(np.reshape(x_test, (-1, 1)), columns=["x"])


def get_distance(x1, x2):
    return exp(-((x1 - x2) ** 2))


def get_neighbors(x, k):
    global df
    df["dist"] = df.apply(lambda row: get_distance(x, row.x), axis=1)
    df = df.sort_values("x", ascending=False).sort_values("dist", ascending=False).drop("dist", 1)
    return df.head(k)


def predict(x, k):
    neighbors = get_neighbors(x, k)
    return 1 / k * (neighbors.y.sum())



def predict_knn(k=1, tau=1):
    """Returns predictions for the values in x_test, using KNN predictor with the specified k."""
    # TODO: your code here
    x_test_df["prediction"] = x_test_df.apply(lambda row: predict(row.x, k), axis=1)
    return x_test_df["prediction"].to_numpy()


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0, 3])

    y_test = predict_knn(k=k)
    # print(y_test)

    plt.scatter(x_train, y_train, label="training data", color="black")
    plt.plot(x_test, y_test, label="predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.savefig("k" + str(k) + ".png")
    plt.show()


for k in (1, 3, len(x_train) - 1):
    plot_knn_preds(k)
