from sklearn.tree import DecisionTreeClassifier


def run(X, y, X_test, max_depth, min_samples_split, max_features):
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, max_features=max_features)
    clf.fit(X, y)
    y_predicted = clf.predict(X_test)
    print(y_predicted)
    return y_predicted


