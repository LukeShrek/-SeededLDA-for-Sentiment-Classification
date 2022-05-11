import datetime

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfModel:
    def __init__(self, ngram):
        self.model = TfidfVectorizer(ngram_range=(1, ngram + 1))

    def train(self, inputs):
        X = [input.stc for input in inputs]
        self.model.fit(X)

    def vectorize(self, inputs):
        X = [input.stc for input in inputs]
        vector_input = self.model.transform(X)
        return vector_input


def feature_select_chi2(stcs, labels, k_best='all', task_name='chi2_score_dict'):
    """
    select top k features through chi-square test
    :param stcs: list of sentences
    :param labels: list of label you want to score chi2 for
    :param k_best:  k words with the best score
    :param label_name: name task that chi2 test for
    :return:
    """
    bin_cv = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, binary=True)
    le = LabelEncoder()
    X = bin_cv.fit_transform(stcs, np.nan)
    y = le.fit_transform(labels).reshape(-1, 1)

    skb = SelectKBest(chi2, k=k_best)
    skb.fit(X, y)

    feature_ids = skb.get_support(indices=True)
    feature_names = bin_cv.get_feature_names()
    result = {}
    vocab = []

    for new_fid, old_fid in enumerate(feature_ids):
        feature_name = feature_names[old_fid]
        vocab.append(feature_name)

    result['word'] = vocab
    result['_score'] = list(skb.scores_)
    result['_pvalue'] = list(skb.pvalues_)
    result_df = pd.DataFrame.from_dict(result)
    result_df = result_df.sort_values('_score', ascending=False).reset_index()
    result_df.to_csv('./data/output/chi2_score_dict/{}.csv'.format(task_name))
    # we only care about the final extracted feature vocabulary
    return result

# import torch
# from transformers import AutoTokenizer, AutoModel
# def phobert_tokenize(inputs, aspectId):
#     phobert = AutoModel.from_pretrained("vinai/phobert-base")
#     tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
#     a = []
#     for t in inputs:
#         input_ids = torch.tensor([tokenizer.encode(t.stc)])
#         with torch.no_grad():
#             features = phobert(input_ids)
#             s = np.array(features[1])
#             # print(s[0])
#             a.append(s[0])
#     return np.array(a)
