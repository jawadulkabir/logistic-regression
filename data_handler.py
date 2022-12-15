import pandas as pd
import numpy as np

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    data = pd.read_csv('data_banknote_authentication.csv')
    ncols = data.shape[1]

    X = data.iloc[:,[i for i in range(0,ncols-1)]]
    y = data.iloc[:,[ncols-1]]

    X = X.to_numpy()
    y = y.to_numpy()

    #add column of ones to X for theta0
    X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
    
    return X, y


def split_dataset(X, y, test_size, shuffle):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    test_set_size = int(test_size*X.shape[0])
    train_set_size = X.shape[0] - test_set_size
    print(train_set_size)
    print(test_set_size)

    #merge X and y to make one matrix before shuffling
    data = np.concatenate((X,y), axis=1)

    if shuffle:
        np.random.shuffle(data)

    ncols = data.shape[1]

    X_train, y_train = data[0:train_set_size,0:ncols-1], data[0:train_set_size,ncols-1]
    X_test, y_test = data[train_set_size:train_set_size+test_set_size, 0:ncols-1], data[train_set_size:train_set_size+test_set_size, ncols-1]

    #make y 2D 
    # y_train = y_train.reshape(y_train.shape[0],1)
    # y_test = y_test.reshape(y_test.shape[0],1)

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
