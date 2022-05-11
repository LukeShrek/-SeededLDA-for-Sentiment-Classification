import datetime
import pickle
from abc import ABC
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from models import PolarityOutput, Input, ComparativeStcOutput
from modules.embeddings import TfidfModel
from modules.models import Model

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class SentimentModel(Model, ABC):
    def __init__(self, vocab_path, vocab_path_lda, model):
        self.vocab_path = vocab_path
        self.vocab_path_lda = vocab_path_lda
        self.vocab = []
        self.model = model
        self.threshold = 0

    def chi2_dict(self, k_best):
        vocab_df = pd.read_csv(self.vocab_path).head(k_best)
        self.vocab = list(vocab_df.word)

    def chi2_dict_lda(self):
        vocab_df = pd.read_csv(self.vocab_path).head(3000)
        self.vocab = list(vocab_df.word)

    def lda_dict(self):
        vocab_df = pd.read_csv(self.vocab_path_lda)
        col = list(vocab_df.columns)[0]
        self.vocab = self.vocab + list(vocab_df[col])

    def chi2_represent(self, inputs):
        """
        :param list of models.Input inputs:
        :return:
        """

        features = []
        for input in inputs:
            _feature = [1 if word in input.stc.split(' ') else 0 for word in self.vocab]
            features.append(_feature)

        return features

    def train(self, encode_inputs: List, outputs: List):
        """
        :param encode_inputs:
        :param outputs:
        """
        X = encode_inputs
        y = outputs
        self.model.fit(X, y)

    def save(self):
        # save the model to disk
        pickle.dump(self.model,
                    open('./data/output/model/comparative_stc/comparative_stc_model{}.pkl'.format(
                        datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), 'wb'))

    def load(self, path):
        # load the model from disk
        model = pickle.load(open(path, 'rb'))

        self.model = model

    def predict(self, encode_inputs: List):
        X = encode_inputs
        if self.threshold != 0:
            predict_prob = self.model.predict_proba(X)[:, 1]
            predict = np.where(predict_prob > self.threshold, 1, 0)
        else:
            predict = self.model.predict(X)
        return predict

    def evaluate(self, y_test, y_predicts, pos_label):
        p = precision_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        r = recall_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        f1 = f1_score(y_test, y_predicts, labels=[pos_label], average = 'macro')
        return p, r, f1

    def evaluate_lda(self, y_test, y_predicts, pos_label=1):
        p = precision_score(y_test, y_predicts, pos_label = pos_label)
        r = recall_score(y_test, y_predicts, pos_label = pos_label)
        f1 = f1_score(y_test, y_predicts, pos_label = pos_label)
        return p, r, f1
