import sklearn.neighbors as neigh



def run(X, y, X_testset, k_value):
    nbrs = neigh.KNeighborsClassifier(n_neighbors=k_value)
    nbrs.fit(X, y)
    y_predicted = nbrs.predict(X_testset)

    return y_predicted
