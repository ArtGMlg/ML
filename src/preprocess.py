import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter

class BasePreprocess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def run(self, ohIncludeBool = False, toDrop: list = []) -> pd.DataFrame:
        self.df.drop(toDrop, axis=1, inplace=True)
        print(f'Found gaps in {list(self.df.isna().any().values).count(True)} columns')
        if list(self.df.isna().any().values).count(True) > 0:
            gc = self.df.isna().any()[self.df.isna().any() == True].index.to_list()
            for i in gc:
                if str(self.df.dtypes.loc[i]) == 'float64':
                    self.df[i].fillna(self.df[i].mean(), inplace=True)
                elif str(self.df.dtypes.loc[i]) == 'int64':
                    self.df[i].fillna(self.df[i].median(), inplace=True)
                elif str(self.df.dtypes.loc[i]) == 'object':
                    self.df[i].fillna(self.df[i].mode()[0], inplace=True)

        print(f'Found {len(list(self.df.dtypes.loc[self.df.dtypes == np.dtype(object)].values))} categorial signs (meaning object types)')
        if len(list(self.df.dtypes.loc[self.df.dtypes == np.dtype(object)].values)) > 0:
            self.df = pd.get_dummies(self.df, columns = self.df.dtypes.loc[self.df.dtypes == np.dtype(object)].index, drop_first=True)
        
        if ohIncludeBool:
            print(f'Found {len(list(self.df.dtypes.loc[self.df.dtypes == np.dtype(bool)].values))} categorial signs (meaning bool types)')
            if len(list(self.df.dtypes.loc[self.df.dtypes == np.dtype(bool)].values)) > 0:
                self.df = pd.get_dummies(self.df, columns = self.df.dtypes.loc[self.df.dtypes == np.dtype(bool)].index, drop_first=True)

        return self.df


class Sampling:
    def sample(X: pd.DataFrame, y: pd.Series, usingMethod: str = 'RandomlyUnderSample'):
        if usingMethod == 'RandomlyOverSample':
            ros = RandomOverSampler(random_state=42)
            X_train_ros, y_train_ros= ros.fit_resample(X, y)
            print(sorted(Counter(y_train_ros).items()))
            return X_train_ros, y_train_ros
        elif usingMethod == 'RandomlyUnderSample':
            rus = RandomUnderSampler(random_state=42)
            X_train_rus, y_train_rus= rus.fit_resample(X, y)
            print(sorted(Counter(y_train_rus).items()))
            return X_train_rus, y_train_rus
        else:
            raise AttributeError(f'Unsupported method {usingMethod}, only RandomlyOverSample and RandomlyUnderSample supported')
