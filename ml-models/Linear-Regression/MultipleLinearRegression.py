import numpy as np

class MultipleLinearRegression:
    def __init__(self, lr=0.0001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
    
    def fit(self, X, y):
        n_samples, features = X.shape
        self.weights = np.zeros((features, 1))
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            loss = (1/n_samples) * np.sum(y_pred-y)**2

            dw = (2/n_samples) * np.dot(X.T, (y_pred-y))
            db = (2/n_samples) * np.sum(y_pred-y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

            self.losses.append(loss)
        return self.weights, self.bias, self.losses
    