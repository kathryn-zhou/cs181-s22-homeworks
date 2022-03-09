import numpy as np
from scipy.stats import multivariate_normal as mvn  # you may find this useful


# Please implement the fit(), predict(), and negative_log_likelihood() methods
# of this class. You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.priors = [0, 0, 0]
        self.means = [0, 0, 0]
        self.covs = 0

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    # TODO: Implement this method!
    def fit(self, X, y):
        for c in y:
            self.priors[c] += 1
        self.priors = np.array(self.priors)/y.shape[0]

        for i in range(len(X)):
            self.means[y[i]] += X[i]
        for j in range(len(self.means)):
            self.means[j] = self.means[j]/(len(X)*self.priors[j])

        if self.is_shared_covariance:
            self.covs = 0
            for i in range(len(X)):
                temp = (X[i] - self.means[y[i]])
                self.covs += np.array([[temp[0]**2, temp[0]*temp[1]], [temp[0]*temp[1], temp[1]**2]])
            self.covs = self.covs/len(X)
        else:
            self.covs = [0, 0, 0]
            for i in range(len(X)):
                temp = (X[i] - self.means[y[i]])
                self.covs[y[i]] += np.array([[temp[0]**2, temp[0]*temp[1]], [temp[0]*temp[1], temp[1]**2]])
            for j in range(len(self.covs)):
                self.covs[j] = self.covs[j]/(len(X)*self.priors[j])

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        preds = []
        for x in X_pred:
            temps = [0, 0, 0]
            for i in range(3):
                if self.is_shared_covariance:
                    result = mvn.pdf(x, self.means[i], self.covs)
                else:
                    result = mvn.pdf(x, self.means[i], self.covs[i])
                temps[i] = result*self.priors[i]
            preds.append(np.array(temps).argmax())
        return np.array(preds)

    # TODO: Implement this method!
    def negative_log_likelihood(self, X, y):
        loss = 0
        if self.is_shared_covariance:
            for i in range(len(X)):
                loss += np.log(self.priors[y[i]]*mvn.pdf(X[i], self.means[y[i]], self.covs))
        else:
            for i in range(len(X)):
                loss += np.log(self.priors[y[i]]*mvn.pdf(X[i], self.means[y[i]], self.covs[y[i]]))
        return -loss