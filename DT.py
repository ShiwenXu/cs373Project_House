from sklearn import tree

def run(X, y, X_validation, Y_validation, k_value):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X, y)
    y_predicted = clf.predict(X_validation)

    return y_predicted

