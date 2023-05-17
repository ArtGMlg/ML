import numpy as np
import math
import pandas as pd
from scipy.stats import norm
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numdifftools as nd
import cv2


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
    

class ClassificationTree:
    def __init__(self, max_depth: int = 3) -> None:
        self.stop = max_depth
        
    def __gini(self, y:pd.Series):
        co = y.value_counts().sort_index()
        alc = sum(y.value_counts().values)
        return sum([i/alc * (1 - i/alc) for i in co.values])
    
    def __whatClass(self, y: pd.Series):
        return y.value_counts().sort_index().apply(lambda x: x/sum(y.value_counts().values)).sort_values().last_valid_index()
        
    def fit(self, X_t:pd.DataFrame, y_t:pd.Series) -> list:
        bool_cols = [col for col in X_t 
                    if np.isin(X_t[col].dropna().unique(), [0, 1]).all()]
        self.scaler = StandardScaler()
        for name in bool_cols:
            X_t[name] = self.scaler.fit_transform(X_t[name].values.reshape(-1, 1), y_t)
            
        dataset = pd.concat([X_t, y_t], axis=1)
        storage = [[dataset]]
        tree = []
        rules = []
        self.r = []
        
        depth = 1
        
        while True:
            temp_storage = []
            rule_store = []
            temp_rule = []
            for leaf in storage[depth-1]:
                better_split = []
                if leaf.shape[1] == 0: 
                    rule_store.append('')
                    temp_rule.append(None, None)
                    continue
                for i in range(leaf.shape[1]-1):
                    gini_for_col = [self.__gini(leaf.iloc[:,leaf.shape[1]-1]), self.__gini(leaf.iloc[:,leaf.shape[1]-1]), 0]
                    # print(gini_for_col)
                    sortedFrame = leaf.sort_values(leaf.iloc[:,i].name)
                    for j in range(sortedFrame.iloc[:,i].shape[0]):
                        gini_l = self.__gini(sortedFrame[0:j].iloc[:,sortedFrame.shape[1]-1])
                        gini_r = self.__gini(sortedFrame[j:sortedFrame.shape[0]-1].iloc[:,sortedFrame.shape[1]-1])
                        # print([gini_l, gini_r])
                        if [gini_l, gini_r] < gini_for_col[0:2] and gini_l != 0 and gini_r !=0:
                            gini_for_col = [gini_l, gini_r, j]
                    
                    gini_for_col.append(leaf.iloc[:,i].name)        
                    better_split.append(gini_for_col)
                
                better_split.sort(key=lambda x: (x[0], x[1]))
                
                best_split = better_split[0]
                
                if leaf.sort_values(best_split[3])[best_split[3]].size == 0:
                    rule_store.append('')
                    temp_rule.append([None, None])
                    continue
                
                rule_store.append(f"If {best_split[3]} <= {leaf.sort_values(best_split[3])[best_split[3]].values[best_split[2]]}. gini = {self.__gini(leaf.iloc[:,leaf.shape[1]-1])}")
                temp_rule.append([best_split[3], 
                                  leaf.sort_values(best_split[3])[best_split[3]].values[best_split[2]],
                                  self.__whatClass(leaf.iloc[:,leaf.shape[1]-1])
                                ])
                
                temp_storage.append(leaf.sort_values(best_split[3])[:best_split[2]])
                temp_storage.append(leaf.sort_values(best_split[3])[best_split[2]:])
                
            storage.append(temp_storage)
            rules.append(rule_store)
            self.r.append(temp_rule)
            
            depth += 1
            
            if depth >= self.stop:
                classes = []
                for res in temp_storage:
                    classes.append(('leaf', self.__whatClass(res.iloc[:,res.shape[1]-1])) if res.size != 0 else '')
                self.r.append(classes)
                break
        
        return rules
                
    def predict(self, X:pd.DataFrame):
        bool_cols = [col for col in X 
                    if np.isin(X[col].dropna().unique(), [0, 1]).all()]
        for name in bool_cols:
            X[name] = self.scaler.transform(X[name].values.reshape(-1, 1))
        
        storage = [[X]]
        
        for index, layer in enumerate(self.r):
            if index + 1 != self.stop:
                temp_st = []
                frames = storage[index]
                for ind, rule in enumerate(layer):
                    sortedFrame = frames[ind].sort_values(rule[0])
                    target = sortedFrame[rule[0]]
                    res_tar = target.reset_index()
                    lastSplitterInd = res_tar.where(res_tar == rule[1]).last_valid_index()
                    if lastSplitterInd == None:
                        lastSplitterInd = res_tar.where(res_tar <= rule[1]).last_valid_index()
                    if lastSplitterInd == None:
                        lastSplitterInd = res_tar.where(res_tar >= rule[1]).last_valid_index()
                    print(sortedFrame[:lastSplitterInd].shape, sortedFrame[lastSplitterInd:].shape, lastSplitterInd, index)
                    temp_st.append(sortedFrame[:lastSplitterInd])
                    temp_st.append(sortedFrame[lastSplitterInd:])
                    
                storage.append(temp_st)
            else:
                frames = storage[index]
                for ind, probableClass in enumerate(layer):
                    if 'leaf' in probableClass:
                        print(frames[ind].shape)
                        frames[ind]['cl'] = [probableClass[1]] * frames[ind].shape[0]
                    else:
                        if ind + 1 % 2 == 0:
                            previousRule = self.r[index - 1][int((ind + 1)/2 - 1)]
                            frames[ind]['cl'] = [previousRule[1]] * frames[ind].shape[0]
                        else:
                            previousRule = self.r[index - 1][int((ind + 2)/2 - 1)]
                            frames[ind]['cl'] = [previousRule[2]] * frames[ind].shape[0]
                            
                print([i.shape for i in frames])
                merged = pd.concat([i for i in frames])
                    
                res = merged.sort_index()
                return res['cl'].values

class ConvNeuroLayer:
    def __init__(self, n_filters: int, filter_size: int, window_size: tuple, stride: int, shape: tuple) -> None:
        self.nf = n_filters
        self.fs = filter_size
        self.ws = window_size
        self.stride = stride
        
        self.kernels = np.zeros((self.nf, self.fs, self.fs))

        for i in range(self.nf):
            self.kernels[i] = np.random.randn(self.fs, self.fs)
            self.kernels[i] /= sum(self.kernels[i])
            
        self.b = np.zeros((shape[0], shape[1]))
        
    def __averagepooling(self, image, kernel_size, stride):
        # Размеры изображения
        input_dim_x, input_dim_y = image.shape
        
        # Размеры окна
        kernel_dim_x, kernel_dim_y = kernel_size
        
        # Вычисляем размерность выходного изображения
        output_dim_x = int((input_dim_x - kernel_dim_x) / stride + 1)
        output_dim_y = int((input_dim_y - kernel_dim_y) / stride + 1)
        
        # Создаем выходной массив
        output = np.zeros((output_dim_x, output_dim_y))
        
        # Применяем операцию пулинга к изображению
        for i in range(output_dim_x):
            for j in range(output_dim_y):
                output[i, j] = np.mean(
                    image[i * stride : i * stride + kernel_dim_x, j * stride : j * stride + kernel_dim_y]
                )
        
        return output
        
    def forward(self, im: np.ndarray) -> np.ndarray:
        filterred = np.zeros((self.nf, im.shape[0], im.shape[1]))
        
        for num, kernel in enumerate(self.kernels):
            filterred[num] = np.sum(cv2.filter2D(im, -1, kernel), axis=2)
            
        filterred += self.b
        
        buf = []
        
        for im in filterred:
            buf.append(self.__averagepooling(im, self.ws, self.stride))
            
        pooled = np.array(buf)
        
        return pooled

class NeuroLayer:
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1+np.exp(-1 * x))
    
    def tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)
    
    def linear(self, x: np.ndarray) -> np.ndarray:
        return x

    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        temp = np.dot(self.connectedToPrevious, self.x) + self.b
        self.a = self.af(temp)
        self.ga = nd.Derivative(self.af)(temp)
        return self.a
    
    def fast_forward(self, x: np.ndarray) -> np.ndarray:
        return self.af(np.dot(self.connectedToPrevious, x) + self.b)
    
    def backward(self, gr: np.ndarray, m, lr) -> np.ndarray:
        buf = np.multiply(self.ga.T, np.dot(gr, m))
        # buf = np.dot(np.multiply(self.ga.T, gr), m)
        self.b -= lr * buf.T
        self.connectedToPrevious -= lr*np.dot(self.x, buf).T
        return buf
    
    def __init__(self, numOfNeurons: int, activation: str) -> None:
        allowedActivation = ['sigmoid', 'tanh', 'relu', 'linear']
        self.n = numOfNeurons
        self.connectedToPrevious = np.array([])
        if activation in allowedActivation:
            if activation == 'sigmoid':
                self.af = self.sigmoid
            elif activation == 'tanh':
                self.af = self.tanh
            elif activation == 'relu':
                self.af = self.relu
            elif activation == 'linear':
                self.af = self.linear
        else:
            raise ValueError(f'Unexpected activation function. Supported functions are: {allowedActivation}')
        
 
class NeuroNet:
    def __init__(self, sequence: list, input_shape: tuple) -> None:
        if not input_shape:
            raise ValueError("You must specify the input shape.")
        self.layers = sequence
        buf = np.random.normal(0, 1, (self.layers[0].n, input_shape[0]))
        self.matrices = [buf]
        self.layers[0].connectedToPrevious = buf
        self.layers[0].b = np.random.normal(0, 1, (self.layers[0].n, 1))
        
    def __MSE(self, yp, yr):
        return np.mean((yr-yp)**2)
    
    def __MAE(self, y_r, y_p):
        return np.mean(abs(y_r-y_p))
    
    def __MAPE(self, y_r, y_p):
        return np.mean(abs(y_r-y_p)/y_r)
        
    def compile(self, loss: str) -> None:
        """
        Случайным образом создаем матрицы весов для перехода от одного слоя к другому
        """
        for i in range(len(self.layers) - 1):
            buf = np.random.normal(0, 1, (self.layers[i+1].n, self.layers[i].n))
            self.matrices.append(buf)
            self.layers[i+1].connectedToPrevious = buf
            self.layers[i+1].b = np.random.normal(0, 1, (self.layers[i+1].n, 1))
            
    def __max_batches(self, n: int) -> int:
        i = n // 2
        m = 20 if n < 100 else 250
        while i > m:
            if n % i == 0:
                return i
            i -= 1
        return i
            
    def fit(self, X:pd.DataFrame, y:pd.Series, e:int, rate: float = 0.01) -> np.ndarray:
        X = X.to_numpy()
        for i in range(e):
            print(f"Initializing epoch {i+1} of {e}")
            
            nbatches = self.__max_batches(len(X))
            
            batchSize = len(X) // nbatches
            
            start = 0
            
            totalp = []
            
            for i in range(batchSize, nbatches, batchSize):
                batch = X[start:i]
                
                for ind, ob in enumerate(batch):
                    ob = ob[np.newaxis, :].T
                    for layer in self.layers:
                        ob = layer.forward(ob)
                    
                    pred = ob
                    totalp.append(pred.flatten()[0])
                    
                    gr = nd.Gradient(self.__MSE)(pred, y.values[start:i][ind])
                    
                    m = 1
                    
                    for layer in self.layers[::-1]:
                        gr = layer.backward(gr, m, rate)
                        m = layer.connectedToPrevious
                        
                print(f"{i}/{nbatches}\t\tLoss: {self.__MSE(totalp, y.values[0:i])}", end='\r')
                
                start += batchSize
                
            print(f"{nbatches}/{nbatches}\t\tLoss: {self.__MSE(totalp, y.values[0:i])}", end='\n')
            
        self.matrices = [layer.connectedToPrevious for layer in self.layers]
                
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        
        X = X.to_numpy()
        pred = []
        for ind, ob in enumerate(X):
            print(f"{ind}/{X.shape[0]}", end='\r')
            ob = ob[np.newaxis, :].T
            for layer in self.layers:
                ob = layer.fast_forward(ob)
            pred.append(ob.flatten()[0])
            
        return np.array(pred)


class kmeans:
    def __init__(self, n_clusters: int) -> None:
        self.n = n_clusters
        self.clusters = [[] for _ in range(n_clusters)]
        self.centroids = []
        
    def __range(self, center, x):
        return np.sqrt(np.sum((center - x)**2))
    
    def __rand_centroids(self, X):
        centroids = np.zeros((self.n, X.shape[1]))
        for i in range(self.n):
            centroids[i] = X[np.random.choice(range(X.shape[0]))]
        self.centroids = centroids
    
    def fit(self, X: np.ndarray) -> None:
        self.__rand_centroids(X)
        while True:
            for i, sample in enumerate(X):
                dist = [self.__range(centroid, sample) for centroid in self.centroids]
                self.clusters[np.argmin(dist)].append(i)
            tmp = self.centroids.copy()
            for i, cluster in enumerate(self.clusters):
                self.centroids[i] = np.mean(X[cluster], axis=0)
            if not (self.centroids - tmp).any() < 1e-8:
                self.labels_ = self.predict(X)
                return
            self.labels_ = self.predict(X)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = []
        for i, sample in enumerate(X):
            dist = [self.__range(centroid, sample) for centroid in self.centroids]
            pred.append(np.argmin(dist))
        return np.array(pred)
        
