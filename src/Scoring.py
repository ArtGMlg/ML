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
    
    def __count_classes(self, y_r, y_p):
        y_r = np.array(y_r)
        av_cl = sorted(list(np.unique(y_r).flatten()))
        classes = {}
        for c in av_cl:
            classes[c] = [0 for i in range(len(av_cl))]

        if len(y_r) != len(y_p):
            raise IndexError(f'Cannot compare objects with shapes {y_r.shape} and {y_p.shape}')

        buf = np.array(y_r)

        for i in range(len(y_r)):
            if buf[i] == y_p[i]:
                classes[buf[i]][av_cl.index(buf[i])] += 1
            else:
                for j in av_cl:
                    if y_p[i] == j:
                        classes[buf[i]][av_cl.index(j)] += 1
                        break

        return classes, av_cl
    
    def Confusion_matrix(self, y_r, y_p):
        classes, _ = self.__count_classes(y_r, y_p)
        buf = []
        for i in classes:
            buf.append(classes[i])

        return np.array(buf)

    def Accuracy(self, y_r, y_p):
        mat = self.Confusion_matrix(y_r, y_p)

        return mat.trace() / mat.sum()


    def Recall(self, y_r, y_p):
        classes, av_cl = self.__count_classes(y_r, y_p)
        buf = {}
        for i in classes:
            buf[i] = classes[i][av_cl.index(i)] / sum(classes[i])

        return buf


    def Precision(self, y_r, y_p):
        classes, av_cl = self.__count_classes(y_r, y_p)
        buf = {}
        for i in classes:
            p = sum([classes[j][av_cl.index(i)] for j in av_cl])
            buf[i] = classes[i][av_cl.index(i)] / p

        return buf
    
    def f1(self, y_r, y_p):
        """Метод подсчета f1"""
        _, av_cl = self.__count_classes(y_r, y_p)

        buf = {}

        pr = self.Precision(y_r, y_p)
        re = self.Recall(y_r, y_p)

        for i in av_cl:
            buf[i] = 1 / ( 1 / (pr[i]) + 1 / (re[i]) )

        return buf
    
    def report(self, y_r, y_p):
        print(f'Accuracy\t{self.Accuracy(y_r, y_p)}\n' + 
              f'Precision\t{self.Precision(y_r, y_p)}\n'+
              f'Recall\t\t{self.Recall(y_r, y_p)}\n'+ 
              f'F1\t\t{self.f1(y_r, y_p)}\n')
