from sklearn.neighbors import NearestNeighbors
import numpy as np

def run(X, y, X_validation, k_value):
    nbrs = NearestNeighbors(n_neighbors=k_value, algorithm='auto').fit(X, y)

    y_predicted = nbrs.predict(X_validation)

    return y_predicted