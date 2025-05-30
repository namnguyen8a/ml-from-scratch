# Implement Linear regression using normal equation

import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        mean_x = np.mean(X)
        mean_y = np.mean(y)

        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)

        self.w = numerator / denominator
        self.b = mean_y - self.w * mean_x
    
    def predict(self, X):
        X = np.array(X)
        return self.w * X + self.b

# Example usage:
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

model = SimpleLinearRegression()
model.fit(X,y)
print(model.w)
print(model.b)
print("Prediction:", model.predict([6, 7]))