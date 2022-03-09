import numpy as np
import pandas as pd
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

# Please implement the predict() method of this class
# You can add additional private methods by beginning them with
# two underscores. It may look like the __dummyPrivateMethod below. You can feel
# free to change any of the class attributes, as long as you do not change any
# of the given function headers (they must take and return the same arguments).

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    # Just to show how to make 'private' methods
    def __dummyPrivateMethod(self, input):
        return None

    def __predict_class(self, mag1, temp1):
        train_df = pd.DataFrame(self.X, columns=["mag", "temp"])
        train_df["y"] = self.y
        train_df["dist"] = train_df.apply(lambda row: ((row["mag"]-mag1)/3)**2 + (row["temp"]-temp1)**2, axis=1)

        train_df = train_df.sort_values("dist", ascending=True).drop("dist", 1)
        return train_df.head(self.K)["y"].mode()[0]

    # TODO: Implement this method!
    def predict(self, X_pred):
        # The code in this method should be removed and replaced! We included it
        # just so that the distribution code is runnable and produces a
        # (currently meaningless) visualization.
        df = pd.DataFrame(X_pred, columns=["mag", "temp"])

        df["class"] = df.apply(lambda row: self.__predict_class(row["mag"], row["temp"]), axis=1)
        return df["class"].to_numpy()

    # In KNN, "fitting" can be as simple as storing the data, so this has been written for you
    # If you'd like to add some preprocessing here without changing the inputs, feel free,
    # but it is completely optional.
    def fit(self, X, y):
        self.X = X
        self.y = y