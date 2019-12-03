import sklearn.neighbors as neigh


def run(X, y, X_validation, k_value):
    nbrs = neigh.KNeighborsClassifier(n_neighbors=k_value)
    nbrs.fit(X, y)
    y_predicted = nbrs.predict(X_validation)

    return y_predicted