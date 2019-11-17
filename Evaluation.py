
def evaluate(labels, predicted_labels):
    size = len(labels)
    loss = 0
    predicted_p_each_range={}
    total_each_range = {}
    true_predicted={}

    for i in range(1,11):
        predicted_p_each_range[i]=0

    for i in range(1,11):
        total_each_range[i]=0

    for i in range(1,11):
        true_predicted[i]=0
    for i in range(size):
        total_each_range[int(labels[i])] +=1
        predicted_p_each_range[int(predicted_labels[i])]+=1
        if int(labels[i]) != int(predicted_labels[i]):
             loss += 1
        if int(predicted_labels[i]) == int(labels[i]):
             true_predicted[int(predicted_labels[i])] += 1
    accuracy = (float(size)-float(loss))/float(size)
    #precision = 0 if true_positive == 0 else (true_positive / predicted_true)
    precision=0
    for i in range(1,11):
        if true_predicted[i] == 0 or predicted_p_each_range[i] == 0:
            precision +=0
            continue
        precision+=(float(true_predicted[i])/float(predicted_p_each_range[i]))
    precision = precision/10
    recall=0
    for i in range(1,11):
        if true_predicted[i] == 0 or total_each_range[i] == 0:
            recall +=0
            continue
        recall+=(float(true_predicted[i])/float(total_each_range[i]))
    recall = recall/10
    f1=0
    if precision+recall == 0:
        f1 == 0
    else:
        f1=(2 * precision * recall / (precision + recall))
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('F-1 score: %f\n' % f1)
    return precision, recall, f1
