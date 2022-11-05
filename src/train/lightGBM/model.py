import numpy as np

# For training
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# lgb
import lightgbm as lgb


class Model(object):
    def __init__(self, params):
        self.params = params

    def preprocessing(self, X, y):
        X_train, self.X_valid, y_train, self.y_valid = train_test_split(X, y)
        self.lgb_train = lgb.Dataset(X_train, y_train)
        self.lgb_eval = lgb.Dataset(self.X_valid, self.y_valid)

    def train(self):
        """train a model"""
        # lightGBMの場合
        self.trained_model = lgb.train(
            **self.params, train_set=self.lgb_train, valid_sets=self.lgb_eval
        )

    def predict(self, X):
        """Predict result"""
        return self.trained_model.predict(X)
