#####################
# CS 181, Spring 2022
# Homework 1, Problem 1
# STARTER CODE
##################

from math import exp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as c

data = [
    (0.0, 0.0),
    (1.0, 0.5),
    (2.0, 1.0),
    (3.0, 2.0),
    (4.0, 1.0),
    (6.0, 1.5),
    (8.0, 0.5),
]
x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])
x_test = np.arange(0, 12, 0.1)
test_inputs = np.reshape(x_test, (-1, 1))


def k(x1, x2, tau):
    return exp(-(x1 - x2) * (x1 - x2) / tau)


def get_prediction(x, tau, infer=True):
    if infer:
        return sum(
            [
                k(data[j][0], x[0], tau) * data[j][1]
                for j in range(len(data))
            ]
        )
    return sum(
        [
            k(data[j][0], x[0], tau) * data[j][1]
            for j in range(len(data))
            if x[0] != data[j][0]
        ]
    )


def compute_loss(tau):
    loss = sum(
        [(data[i][1] - get_prediction(data[i], tau, infer=False)) ** 2 for i in range(len(data))]
    )
    print("Loss for tau = " + str(tau) + ": " + str(loss))
    return loss

for tau in (0.01, 2, 100):
    y_test = np.apply_along_axis(get_prediction, 1, test_inputs, tau=tau, infer=True)
    plt.scatter(x_train, y_train, label="training data", color="black")
    plt.plot(x_test, y_test, label="predictions using tau = " + str(tau))
    plt.legend()
    plt.title("Predictions with tau = " + str(tau))
    plt.savefig("tau" + str(tau) + ".png")
    plt.show()

    compute_loss(tau)
