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

    tp = np.count_nonzero(np.logical_and(y_pred == y_true, y_pred == 1))
    pp = np.count_nonzero(y_pred == 1)
    return tp/pp


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement

    tp = np.count_nonzero(np.logical_and(y_pred == y_true, y_true == 1))
    p = np.count_nonzero(y_true == 1)
    return tp/p


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
