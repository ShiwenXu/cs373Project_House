from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def run(X, y, X_validation, k_value):
    nbrs = KNeighborsClassifier(n_neighbors=k_value)
    nbrs.fit(X, y)
    y_predicted = nbrs.predict(X_validation)

    return y_predicted