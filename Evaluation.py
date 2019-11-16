
def evaluate(true_labels, predicted_labels):
    size = len(true_labels)
    predicted_p = 0
    total_p = 0
    true_p = 0
    for i in range(size):
        if predicted_labels[i]:
            predicted_p += 1
            if true_labels[i]:
                true_p += 1
        if true_labels[i]:
            total_p += 1

    precision = 0 if true_p == 0 else (true_p / predicted_p)
    recall = 0 if true_p == 0 else (true_p / total_p)
    f1 = 0 if true_p == 0 else (2 * precision * recall / (precision + recall))
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F-1 score: %f' % f1)
    return precision, recall, f1