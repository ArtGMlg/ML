import numpy as np
import math
import pandas as pd
from scipy.stats import norm


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


class kNN:
    def __init__(self, X, y, k=2):
        self.X_train = X.values
        self.y_train = y.values
        self.k = k
        self.nc = len(np.unique(y.values))
        self.classes = list(np.unique(y.values).flatten())

    def __dist(self, p, q):
        return math.sqrt(np.sum((p-q)**2))

    def predict(self, Xt) -> np.ndarray:
        self.Xt = Xt.values
        prediction = []
        counter = 0
        for item in self.Xt:
            distances = [(self.__dist(item, self.X_train[i]), self.y_train[i]) for i in range(len(self.X_train))]
            # distances.sort(key=lambda x: x[0])
            av_cl = {}
            for i in self.classes:
                av_cl[i] = 0
            for d in sorted(distances, key=lambda x: x[0])[0: self.k]:
                av_cl[d[1]] += 1
            prediction.append(list(dict(sorted(av_cl.items(), key=lambda item: item[1])).items())[0][0])
            counter += 1

        return np.array(prediction)
    

class NaiveBayes:
    def fit(self,  X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y
        self.means = self.X.groupby(self.y).mean()
        self.stds = self.X.groupby(self.y).apply(np.std)
        self.probs = self.X.groupby(self.y).apply(lambda x: len(x)) / self.X.shape[0]

    def predict(self, Xt: pd.DataFrame) -> np.ndarray:
        y_pred = []
        for i in range(Xt.shape[0]):
            p = {}

            for cl in np.unique(self.y):
                p[cl] = self.probs.iloc[[cl]].values[0]
                for index, param in enumerate(Xt.iloc[i]):
                    p[cl] *= norm.pdf(param,
                                      self.means.iloc[[cl]].iloc[:, index].values[0],
                                      self.stds.iloc[[cl]].iloc[:, index].values[0]
                                      )

            y_pred.append(pd.Series(p).values.argmax())

        return np.array(y_pred)
    

class CART:
    
