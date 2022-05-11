import pandas as pd
import numpy as np
import time
import re
from pprint import pprint
import joblib
import sys

sys.path.insert(1, './scripts/')
import os
from sklearn.feature_extraction.text import CountVectorizer
import guidedlda

# dirname = os.path.dirname(__file__)
# path = os.path.join(dirname, 'data/input/mebe_tiki.csv')
df = pd.read_csv(
    open("C:/Users/lemin/PycharmProjects/kltn/Opinion-Mining/data/raw_data/mebe_tiki.csv", 'r', encoding='utf-8'))
symbols = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.',
           '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '“', '']

inputs = []
vocab = []


def preprocessing():
    # Data cleaning:
    for _, r in df.iterrows():
        t = str(r['text']).split()
        string = ''
        for s in t:
            s = re.sub(r'\d+', '', s)
            if s not in symbols and len(s) >= 2:
                # # print(s)
                vocab.append(s)
                string += s + ' '

        inputs.append(string[:-1])


preprocessing()
vocab = list(dict.fromkeys(vocab))

token_vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=vocab)
# print(token_vectorizer)
X = token_vectorizer.fit_transform(inputs)
word2id = token_vectorizer.vocabulary_

ship = ['đóng_gói', 'giao_hàng', 'giao', 'hộp', 'vận_chuyển', 'nhanh', 'dự kiến']
price = ['giá', 'rẻ', 'giá_thành', 'deal', 'sale', 'tiền']
quality = ['chất_lượng', 'chất_liệu', 'mùi', 'xài', 'thấm hút']
safety = ['date', 'hạn', 'hsd', 'vệ_sinh', 'sạch_sẽ', 'yên_tâm']
service = ['cskh', 'thân_thiện', 'phản_hồi', 'trả_lời', 'tặng', 'khuyến_mại', 'dịch_vụ', 'quà', 'quà_tặng', 'liên_hệ',
           'chương_trình', 'nhân_viên']
genuineness = ['mã_vạch', 'code', 'mã', 'nhãn_mác']

ship = [x for x in ship if x in list(word2id.keys())]
price = [x for x in price if x in list(word2id.keys())]
quality = [x for x in quality if x in list(word2id.keys())]
safety = [x for x in safety if x in list(word2id.keys())]
service = [x for x in service if x in list(word2id.keys())]
genuineness = [x for x in genuineness if x in list(word2id.keys())]

print(type(ship))
model = guidedlda.GuidedLDA(n_topics=6, n_iter=100, random_state=10, refresh=20)
seed_topic_list = [ship, price, quality, safety, service, genuineness]
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

model.fit(X.toarray().astype(int), seed_topics=seed_topics, seed_confidence=0.15)
joblib.dump(model, "Guided_LDA_6topics.pkl")
n_top_words = 30
topic_word = model.topic_word_
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
