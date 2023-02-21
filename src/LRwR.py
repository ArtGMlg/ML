import numpy as np


class LRwR:
    def grQ(self, w, n, alpha):
        buf = [0 for i in range(len(w))]
        for i in range(len(self.MyX)):
            buf += (self.MyX[i] * w - self.y.values[i])*self.MyX[i][n] + alpha*1000*np.sign(w)
        return buf

    def fit(self, X, y, k=0.00000001, alpha=0.5):
        self.y = y
        self.MyX = np.concatenate((np.ones((X.values.shape[0], 1), dtype=np.int64), X.values), axis=1)
        self.w = np.array([0 for i in range(len(self.MyX[0]))], dtype=np.float64)
        for i in range(len(self.w)):
            self.w -= (k/len(self.w))*self.grQ(self.w, i, alpha)
    
    def predict(self, X):
        return np.sum(X.values * self.w[1:len(self.w)], axis=1)
