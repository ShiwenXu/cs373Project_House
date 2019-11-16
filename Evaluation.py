
def evaluate(labels, predicted_labels):
    size = len(labels)
    err = 0
    predicted_positive = 0
    total_positive = 0
    true_positive = 0
    for i in range(size):
        if labels[i]:
            total_positive += 1
        if predicted_labels[i]:
            predicted_positive += 1
            if labels[i]:
                true_positive += 1
        if labels[i] != predicted_labels[i]:
            err += 1
    accuracy = (size - err) / size
    precision = 0 if true_positive == 0 else (true_positive / predicted_positive)
    recall = 0 if true_positive == 0 else (true_positive / total_positive)
    f1 = 0 if true_positive == 0 else (2 * precision * recall / (precision + recall))
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F-1 score: %f' % f1)
    return accuracy, precision, recall, f1