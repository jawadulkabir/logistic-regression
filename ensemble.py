from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator, params):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        #initialize base_estimator and n_estimator
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.params = params


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        #fit the base_estimator
        self.estimators = []

        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            estimator = self.base_estimator(self.params)
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        #predict the base_estimator using majority voting
        y_pred = np.zeros((X.shape[0],self.n_estimator))
        for i,estimator in enumerate(self.estimators):
            y_pred[:,i]= estimator.predict(X).reshape(-1)
        
        y_pred = y_pred.astype(int)
        y_pred = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 1, y_pred) #majority voting

        return y_pred

        