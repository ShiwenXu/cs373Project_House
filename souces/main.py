import tune_hyperparameter
import KNN
import preprocess
import DT
import Evaluation
import numpy as np


def main():
    #data, labels, data_600, label_600, test_data, test_price = preprocess.obtain_result()
    # return from process data:
    data_500_tune, labels_500_tune,data_600, label_600, data_1000,label_1000,test_data,test_label=preprocess.obtain_result()
    best_k,best_max_depth,best_min_samples_split,best_max_feature=tune_hyperparameter.tuning(data_500_tune , labels_500_tune)
    data_600=np.asarray(data_600.astype(np.float))
    label_600=np.asarray(label_600.astype(np.float))
    data_1000=np.asarray(data_1000.astype(np.float))
    label_1000=np.asarray(label_1000.astype(np.float))
    test_data=np.asarray(test_data.astype(np.float))
    test_label=np.asarray(test_label.astype(np.float))
    #test on 600 samples:
    knn_pred_600=KNN.run(data_600, label_600, test_data, best_k)
    knn_pred_600=(np.asmatrix(knn_pred_600)).T
    # test on 1000 samples:
    knn_pred_1000=KNN.run(data_1000, label_1000, test_data, best_k)
    knn_pred_1000=(np.asmatrix(knn_pred_1000)).T

    dt_pre_600=DT.run(data_600,label_600,test_data,best_max_depth,best_min_samples_split,best_max_feature,4)
    dt_pre_600= (np.asmatrix(dt_pre_600)).T
    # print(dt_pre_600)
    dt_pre_1000=DT.run(data_1000,label_1000,test_data,best_max_depth,best_min_samples_split,best_max_feature,4)
    dt_pre_1000=(np.asmatrix(dt_pre_1000)).T
    print("KNN_eval_600:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, knn_pred_600)
    print("KNN_eval_1000:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, knn_pred_1000)
    print("DT_eval_600:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, dt_pre_600)
    print("DT_eval_1000:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, dt_pre_1000)


main()