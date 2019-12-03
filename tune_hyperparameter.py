import numpy  as np
import k_fold
import preprocess

#tuning hyperparameter 500 data,
# test with 600 data and 1000 data to show the accuracy of algorithm and ROV curve

def tuning(data_500_tune, labels_500_tune):
    # tuning the best k for K NEAREST NEIGHBOR:
    max_f1=-1
    best_k_nn=-1
    for max_k in range(2, 12):
        print('max_k=%d:' % max_k)
        f1 = k_fold.k_fold_cv(data_500_tune,labels_500_tune,max_k,0)
        if f1 > max_f1:
            max_f1 = f1
            best_k_nn = max_k
    print('Best max_k with 600 data: %d\n' % best_k_nn)

    #tuning the best max depth:
    max_f1 = -1
    best_max_depth = -1
    for max_depth in range(2, 8):
        print('max_depth=%d:' % max_depth)
        f1 = k_fold.k_fold_cv(data_500_tune, labels_500_tune,max_depth,1)
        if f1 > max_f1:
            max_f1 = f1
            best_max_depth = max_depth
    print('Best max_depth: %d\n' % best_max_depth)

    #tuning the min_sample_split:
    max_f1 = -1
    best_min_samples_split = -1
    for min_samples_split in range(2, 10):
        print('min_samples_split=%d:' % min_samples_split)
        f1 =k_fold.k_fold_cv(data_500_tune, labels_500_tune,min_samples_split,2)
        if f1 > max_f1:
            max_f1 = f1
            best_min_samples_split = min_samples_split
    print('Best min_samples_split: %d\n' % best_min_samples_split)


    #tuning the max feature:
    max_f1 = -1
    best_max_feature = 0
    for max_feature in range(2, 10):
        print('max_feature=%d:' % max_feature)
        f1 = k_fold.k_fold_cv(data_500_tune, labels_500_tune,max_feature,3)
        if f1 > max_f1:
            max_f1 = f1
            best_max_feature = min_samples_split
    print('Best max_feature: %d\n' % max_feature)

    return best_k_nn,best_max_depth,best_min_samples_split,best_max_feature



