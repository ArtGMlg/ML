import numpy as np
import math
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


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

            
        
            
        
