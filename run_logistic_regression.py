"""
main code that you will run
"""

from linear_model import LogisticRegression
from data_handler import load_dataset, split_dataset
from metrics import accuracy, precision_score, recall_score, f1_score


if __name__ == '__main__':
    # data load
    X, y = load_dataset()

    # split train and test
    X_train, y_train, X_test, y_test = split_dataset(X, y, 0.3, True)

    # training
    params = dict(
        learning_rate=0.05,
        n_iter=1000,
        batch_size=32,
        n_features=X_train.shape[1],
    )

    classifier = LogisticRegression(params)

    # training
    classifier.fit(X_train, y_train)

    # testing
    y_pred = classifier.predict(X_test)

    # performance on test set
    print('Accuracy ', accuracy(y_true=y_test, y_pred=y_pred))
    print('Recall score ', recall_score(y_true=y_test, y_pred=y_pred))
    print('Precision score ', precision_score(y_true=y_test, y_pred=y_pred))
    print('F1 score ', f1_score(y_true=y_test, y_pred=y_pred))
