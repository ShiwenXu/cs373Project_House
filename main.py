import tune_hyperparameter
import KNN
import preprocess
import DT
import Evaluation


def main():
    #data, labels, data_600, label_600, test_data, test_price = preprocess.obtain_result()
    # return from process data:
    data_500_tune, labels_500_tune,data_600, label_600, data_1000,label_1000,test_data,test_label=preprocess.obtain_result()
    best_k,best_max_depth,best_min_samples_split,best_max_feature=tune_hyperparameter.tuning(data_500_tune , labels_500_tune)
    #run test data and draw roc curve on both KNN and decision tree
    knn_pred_600=KNN.run(data_600, label_600, test_data, best_k)
    knn_pred_1000=KNN.run(data_1000, label_1000, test_data, best_k)
    dt_pre_600=DT.run(data_600,label_600,test_data,best_max_depth,best_min_samples_split,best_max_feature)
    dt_pre_1000=DT.run(data_1000,label_1000,test_data,best_max_depth,best_min_samples_split,best_max_feature)
    print("KNN_eval_600:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, knn_pred_600)
    print("KNN_eval_1000:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, knn_pred_1000)
    print("DT_eval_600:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, dt_pre_600)
    print("DT_eval_1000:\n")
    precision, recall, f1 = Evaluation.evaluate(test_label, dt_pre_1000)