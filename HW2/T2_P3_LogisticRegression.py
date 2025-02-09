import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

# Please implement the fit(), predict(), and visualize_loss() methods of this
# class. You can add additional private methods by beginning them with two
# underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.runs = 200000
        self.neglikeloss = []

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __softmax(self, X):
        z = np.dot(X, self.W.T) 
        return np.exp(z - np.expand_dims(scipy.special.logsumexp(z, axis=1), 1))
    
    # TODO: Implement this method!
    def fit(self, X, y):
        # Add bias
        X = np.column_stack([np.ones(X.shape[0]), X]) 
        # One hot encoding
        y = np.array(pd.get_dummies(y))

        # Generate random weights
        self.W = np.random.rand(3, X.shape[1])

        # Gradient descent
        for _ in range(self.runs):
            grad = np.dot((self.__softmax(X)-y).T,X) 
            self.W -= self.eta * (grad + 2 * self.lam * self.W)
            loss = -np.sum(y*np.log(self.__softmax(X))) + self.lam * (self.W ** 2).sum()
            self.neglikeloss.append(loss)
            
    # TODO: Implement this method!
    def predict(self, X_pred):
        # # The code in this method should be removed and replaced! We included it
        # # just so that the distribution code is runnable and produces a
        # # (currently meaningless) visualization.
        X_pred = np.column_stack([np.ones(X_pred.shape[0]), X_pred])
        return np.argmax(np.dot(X_pred, self.W.T), axis=1)

    # TODO: Implement this method!
    def visualize_loss(self, output_file, show_charts=False):
        plt.plot(self.neglikeloss)
        plt.title("Loss over Iterations, eta = " + str(self.eta) + ", lam = " + str(self.lam))
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.title(output_file)
        plt.savefig(output_file)
        if show_charts:
            plt.show()
