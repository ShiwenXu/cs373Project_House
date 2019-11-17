
def evaluate(labels, predicted_labels):
    size = len(labels)
    loss = 0
    predicted_p = 0
    total_p = 0
    true_positive = 0
    for i in range(size):

        if predicted_labels[i]:
            predicted_p += 1
            if labels[i]:
                true_positive += 1
        if labels[i]:
            total_p += 1
        if labels[i] != predicted_labels[i]:
            loss += 1
    accuracy = (size - loss) / size
    precision = 0 if true_positive == 0 else (true_positive / predicted_p)
    recall = 0 if true_positive == 0 else (true_positive / total_p)
    f1 = 0 if true_positive == 0 else (2 * precision * recall / (precision + recall))
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F-1 score: %f' % f1)
    return precision, recall, f1