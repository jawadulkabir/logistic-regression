import numpy as np

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        # print(params)
        # for x in params:
        #     print(x,type(x),type(params[x]),end="\n\n\n")

        self.learning_rate = params['learning_rate']
        self.n_iter = params['n_iter']
        self.n_features = params['n_features']

        self.theta = np.random.randn(self.n_features)
        print(self.theta)

    def sigmoid(self, z):
        """
        :param z:
        :return:
        """
        return 1 / (1 + np.exp(-z))
    

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0], "X and y should have same number of rows"
        assert len(X.shape) == 2, "X should be 2D"
        
        # todo: implement

        train_set_size = X.shape[0]


        for epoch in range(self.n_iter):
            loss = 0
            for i in range(train_set_size):
                x = X[i]
                h = self.sigmoid(np.dot(x, self.theta))
                gradient = np.dot((y[i]-h), x)
                self.theta = self.theta + self.learning_rate * gradient
                try:
                    loss += -y[i]*np.log(h,where=h>0) - (1-y[i])*np.log(1-h,where=1-h>0)
                except:
                    print("epoch ",epoch," RuntimeWarning ", "h: ")
            
            if epoch%5==0:
                print("Epoch: {}, Loss: {}".format(epoch, loss))
                #print(self.theta)   

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        y_pred = np.dot(X,self.theta)
        y_pred = self.sigmoid(y_pred)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        return y_pred




# x = X[i].reshape(n_features,1)
#                 y_hat = self.sigmoid(np.dot(theta.T,x))
#                 error = y[i] - y_hat
#                 gradient = np.dot(x,error)
#                 theta = theta + self.learning_rate*gradient

# what are the things to keep in mind while implementing logistic regression from scratch in python?