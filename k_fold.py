import numpy as np
import math
import KNN
import Evaluation
import DT

k = 5


def k_fold_cv(training_data, training_label, k_value,algo):
    # Your code goes here
    (n, d) = np.shape(training_data)
    f1_sum = 0
    for i in range(k):
        T = []
        S = []
        start = int(math.floor((n * i) / k))
        end = int(math.floor((n * (i + 1)) / k - 1))

        for j in range(n):
            S.append(j)

        for j in range(start, end + 1):
            T.append(j)

        for j in T:
            if j in S:
                S.remove(j)

        X_train = np.asarray(training_data[S[0]:S[-1]:]).astype(np.float)
        y_train = np.asarray(training_label[S[0]:S[-1]:]).astype(np.float)
        # y_train=np.asarray(y_train)
        # y_train=np.reshape(y_train,(1,len(y_train)))
        y_train = y_train.ravel()
        # print(type(y_train))

        X_validation = training_data[T[0]:T[-1]:].astype(np.float)
        y_validation = training_label[T[0]:T[-1]:].astype(np.float)
        y_predict = 0
        if algo == 0:
            y_predicted = KNN.run(X_train, y_train, X_validation, k_value)
            # print y_predicted
        # max_depth
        elif algo == 1:
            y_predicted = DT.run(X_train, y_train, X_validation, k_value, 0, 0, 1)
            print(y_predicted)
        # min_split
        elif algo == 2:
            y_predicted = DT.run(X_train, y_train, X_validation, 0, k_value, 0, 2)
        # max_feature
        elif algo == 3:
            y_predicted = DT.run(X_train, y_train, X_validation, 0, 0, k_value, 3)

        precision, recall, f1 = Evaluation.evaluate(y_validation, y_predicted)
        f1_sum += f1
        return f1_sum / k
