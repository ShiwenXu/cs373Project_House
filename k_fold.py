import numpy as np
import math
import KNN
import Evaluation

k = 5

def k_fold_cv(training_data, training_label, k_value, algo):
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
        #y_train=np.asarray(y_train)
        #y_train=np.reshape(y_train,(1,len(y_train)))
        y_train=y_train.ravel()
       # print(type(y_train))

        X_validation = training_data[T[0]:T[-1]:].astype(np.float)
        y_validation = training_label[T[0]:T[-1]:].astype(np.float)

        if algo == 0:
            y_predicted = KNN.run(X_train, y_train, X_validation,y_validation, k_value)
            # print y_predicted
            precision, recall, f1 = Evaluation.evaluate(y_validation, y_predicted)
            f1_sum += f1
            return f1_sum / k

