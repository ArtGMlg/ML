import numpy as np
from math import sqrt


class Scoring:
    def MAE(self, y_r, y_p):
        return np.mean(abs(y_r-y_p))

    def MSE(self, y_r, y_p):
        return np.mean((y_r-y_p)**2)

    def RMSE(self, y_r, y_p):
        return sqrt(np.mean((y_r-y_p)**2))

    def MAPE(self, y_r, y_p):
        return np.mean(abs(y_r-y_p)/y_r)

    def score(self, y_r, y_p):
        return 1 - (np.sum((y_p-np.mean(y_r))**2))/(np.sum((y_r-np.mean(y_r))**2))

