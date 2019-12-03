from sklearn.tree import DecisionTreeClassifier
import numpy as np

def run(X, y, X_test, var, flag):
    print (np.shape(X))
    print (np.shape(y))
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


