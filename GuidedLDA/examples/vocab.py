import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

data = pd.read_csv("C:/Users/lemin/PycharmProjects/kltn/Opinion-Mining/data/raw_data/tech_tiki.csv")
data["text"].str.lower()

print(type(data["text"]))
# tokenized_sents = [word_tokenize(i) for i in data["text"]]
# print(tokenized_sents)
#
# flattened = []
# for sublist in tokenized_sents:
#     for val in sublist:
#         flattened.append(val)
# print(flattened)
#
# vocab = []
# for item in flattened:
#     if not item in vocab:
#         vocab.append(item)
# print(vocab)