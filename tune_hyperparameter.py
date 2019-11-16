import numpy  as np
import k_fold
import preprocess

def tuning():
    #tuning the best k for K NEAREST NEIGHBOR:
    data,labels,data_600,label_600=preprocess.obtain_result()
    max_f1=-1
    best_k_nn=-1
    for max_depth in range(1, 8):
        print('max_depth=%d:' % max_depth)
        f1 = k_fold(np.data.T, labels,0)
        if f1 > max_f1:
            max_f1 = f1
            best_max_depth = max_depth
    print('Best max_depth: %d\n' % best_max_depth)


    #tuning the best max depth:
    max_f1 = -1
    best_max_depth = -1
    for max_depth in range(1, 6):
        print('max_depth=%d:' % max_depth)
        f1 = k_fold(data, labels,1)
        if f1 > max_f1:
            max_f1 = f1
            best_max_depth = max_depth
    print('Best max_depth: %d\n' % best_max_depth)

    #tuning the min_sample_leaf:
    max_f1 = -1
    best_min_samples_split = -1
    for min_samples_split in range(2, 10):
        print('min_samples_split=%d:' % min_samples_split)
        f1 = k_fold(data, labels,1)
        if f1 > max_f1:
            max_f1 = f1
            best_min_samples_split = min_samples_split
    print('Best min_samples_split: %d\n' % best_min_samples_split)


    #tuning the max feature:
    max_f1 = -1
    best_max_feature = 0
    for max_feature in range(2, 10):
        print('max_feature=%d:' % max_feature)
        f1 = k_fold(data, labels,1)
        if f1 > max_f1:
            max_f1 = f1
            best_max_feature = min_samples_split
    print('Best max_feature: %d\n' % max_feature)

tuning()


