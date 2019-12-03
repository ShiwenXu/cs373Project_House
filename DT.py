from sklearn import tree
import preprocess


def run():
    # X, y, X_validation, k_value
    data_1000, lables_1000, data_600, labels_600, test_data, test_price = preprocess.obtain_result()
    print(labels_600)
    clf = tree.DecisionTreeClassifier()
    clf.fit(data_600, labels_600)
    y_predicted = clf.predict(test_data)
    print(y_predicted)
    return y_predicted


run()
