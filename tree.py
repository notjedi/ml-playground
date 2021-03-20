import pandas as pd
import numpy as np


class Node():

    """A class to represent each node in the Decision Tree"""

    def __init__(self, gini):
        self.gini = gini
        self.threshold = 0
        self.left = None
        self.right = None


class DecisionTreeClassifer():

    def __init__(self, estimators):
        self.n_estimators_ = estimators
        self.features_ = None
        self.classes_ = None
        self.n_features_ = 0
        self.n_classes_ = 0
        self.n_samples_ = 0

    def best_split(self, X, y):
        """Fine best split for a node from all features"""

        classes_root = [np.sum(y == i) for i in self.classes_]
        best_gini = 1 - (np.sum(n / self.classes_) for n in classes_root)
        best_index = 0

        for feature in range(self.features_):

            # TODO sorted(X, y) are not unpacked correctly hence the shapes are different
            x_sorted, y_sorted = sorted(zip(X[:, feature], y))
            classes_right = classes_root.copy()
            classes_left = [0] * self.n_classes_

            for i in range(1, self.n_samples_):

                c = y_sorted[i-1]
                classes_left[c] += 1
                classes_right[c] -= 1

                gini_left = 1 - \
                    (np.sum((classes_left[x] / i) ** 2)
                     for x in self.classes_)
                gini_right = 1 - \
                    (np.sum((classes_right[x] / (self.n_samples_ - i)) ** 2)
                     for x in self.classes_)

                gini = (i * gini_left + (self.n_samples_ - i)
                        * gini_right) / self.n_samples_

                if (x_sorted[i] == x_sorted[i+1]):
                    continue

                best_gini = gini
                best_index = i
                best_split = (x_sorted[i] + x_sorted[i-1]) / 2

                return best_gini, best_index, best_split

    def _build_tree(self):
        """Build the Decision Tree recursively"""
        pass

    def fit(self, X, y):
        """Fit the input"""
        self.features_ = X.index()
        self.classes_ = set(y)
        self.n_classes_ = len(self.classes_)
        self.n_samples_, self.n_features_ = X.shape
        self._build_tree()

    def predict(self):
        """Predict the output"""
        pass


if __name__ == "__main__":
    pass
