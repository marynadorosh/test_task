import os
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, SGDClassifier


class Model:
    def __init__(self, data_path, model_type=SGDClassifier):

        self.data_path = data_path
        self.X = np.load(os.path.join(self.data_path, 'X.npy'))
        self.y = np.load(os.path.join(self.data_path, 'Y.npy'))
        self.model_type = model_type
        self.model = None
        self.predictions = None


    def create_model(self, **params):

        self.model = self.model_type(**params)


    def train(self):
        
        self.model.fit(self.X, self.y)
        self.model.score(self.X, self.y)
    


    def predict(self, test_path):
        test = np.load(os.path.join(test_path, 'X.npy'))
        clf.predictions = self.model.predict(test)
    


    def save_predictions(self, text):
        (pd.DataFrame({'text': text,
                      'preds':clf.predictions})
                      .to_csv('testSet.csv', index=False, header=False))
