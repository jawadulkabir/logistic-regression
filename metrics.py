"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import numpy as np
def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    count = np.count_nonzero(y_true == y_pred)    
    return count/len(y_true)



def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #count number of true positives
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            count += 1
    return count/np.count_nonzero(y_pred == 1)


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #count number of true positives
    count = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            count += 1
    return count/np.count_nonzero(y_true == 1)


def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2*precision*recall/(precision+recall)
