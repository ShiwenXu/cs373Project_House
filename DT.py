from sklearn.tree import DecisionTreeClassifier


def run(X, y, X_test, var, flag):
    if flag == 1:
        clf = DecisionTreeClassifier(max_depth=var)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    elif flag == 2:
        clf = DecisionTreeClassifier(min_samples_split=var)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    elif flag == 3:
        clf = DecisionTreeClassifier(max_features=var)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    return y_predicted


