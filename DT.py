from sklearn.tree import DecisionTreeClassifier
import numpy as np

def run(X, y, X_test, best_max_depth, best_min_samples_split, best_max_feature, flag):
    if flag == 1:
        clf = DecisionTreeClassifier(max_depth=best_max_depth)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    elif flag == 2:
        clf = DecisionTreeClassifier(min_samples_split=best_min_samples_split)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    elif flag == 3:
        clf = DecisionTreeClassifier(max_features=best_max_feature)
        clf.fit(X, y)
        y_predicted = clf.predict(X_test)
        print(y_predicted)
    return y_predicted


