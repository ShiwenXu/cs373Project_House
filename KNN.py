from sklearn.neighbors.nearest_centroid import NearestCentroid
import sklearn.neighbors as neigh
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def run(X, y, X_testset, k_value):
    nbrs = neigh.KNeighborsClassifier(n_neighbors=k_value)
    nbrs.fit(X, y)
    y_predicted = nbrs.predict(X_testset)
    return y_predicted