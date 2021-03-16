import numpy as np
from math import sqrt
from collections import Counter

class KNNClassifier:
    
    def __init__(self, k):
        """初始化KNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        self._X_train = None
        self._y_train = None
    
    def fit(self, X_train, y_train):
        """根据训练集X_train， y_train训练KNN分类器"""
        self._X_train = X_train
        self._y_train = y_train
        return self
    
    def predict(self, X_predict):
        """给定待预测数据集X_predict，返回表示的结果向量"""
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)
    
    def _predict(self, x):
        """给定单独数据X，返回X的预测结果"""
        distances = [sqrt(np.sum(x_train - x)**2) for x_train in self._X_train]
        nearest = np.argsort(distances)
        topk_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topk_y)
        return votes.most_common(1)[0][0]

    def score(self, X_text, y_text):
        y_predict = self.predict(X_text)
        print(sum(y_predict==y_text))/len(y_text)
    
    def __repr__(self):
        return "KNN(k=%d)" % self.k