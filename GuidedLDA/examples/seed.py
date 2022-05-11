import pandas as pd
import numpy as np
import time
import re
from pprint import pprint
import joblib
import sys
import csv
import os
from sklearn.feature_extraction.text import CountVectorizer
import guidedlda


sys.path.insert(1, './scripts/')

# domain = 'mebe'
domain = 'tech'
# source = 'tiki'
# source = 'shopee'
source = 'all'

dirname = 'C:/Users/lemin/PycharmProjects/kltn/Opinion-Mining/'
filename = 'data/raw_data/{}_{}.csv'.format(domain, source)
data_path = os.path.join(dirname, filename)

ASPECT = {
    'mebe': 'price,service,safety,quality,delivery,authenticity',
    'tech': 'price,service,delivery,performance,authenticity,hardware,accessories,appearance'
}
aspect = ASPECT[domain].split(',')
print(aspect)

data_df = pd.read_csv(open(data_path, 'r', encoding='utf-8'))
symbols = ['~', ':', "'", '+', '[', '\\', '@', '^', '{', '%', '(', '-', '"', '*', '|', ',', '&', '<', '`', '}', '.',
           '=', ']', '!', '>', ';', '?', '#', '$', ')', '/', '“', '']
confused = ['không', 'chưa', 'cẩn', 'hài', 'ok', 'hài_lòng', 'sp', 'kì', 'sản phẩm', 'sản_phẩm', 'mua', 'máy', 'điện_thoại',
            'hàng', 'bỉm', 'sữa', 'tả', 'tã', 'bé', 'tiki', 'shopee', 'mẹ', 'bobby', 'đầy đủ', 'đủ dùng', 'tốt', 'được', 'có',
            'thấy', 'nhưng', 'là', 'cho', 'mình', 'như', 'mà', 'rất', 'hơi']

inputs = []
vocab = []


def preprocessing():
    # Data cleaning:
    for _, r in data_df.iterrows():
        t = str(r['text']).split()
        string = ''
        for s in t:
            s = re.sub(r'\d+', '', s)
            if s not in symbols and s not in confused and len(s) >= 1:
                # # print(s)
                vocab.append(s)
                string += s + ' '

        inputs.append(string[:-1])


preprocessing()
vocab = list(dict.fromkeys(vocab))

token_vectorizer = CountVectorizer(ngram_range=(1, 3), vocabulary=vocab)
# print(token_vectorizer)
X = token_vectorizer.fit_transform(inputs)
word2id = token_vectorizer.vocabulary_

seed_round = 5

seedname = 'data/vocab/lda_seed/total_seed{}/{}_seed{}.csv'.format(seed_round-1, domain, seed_round-1)
seedpath = os.path.join(dirname, seedname)
seed_dict_df = pd.read_csv(open(seedpath, 'r', encoding='utf-8'))

# for a in aspect:
#     aspect[a] = seed_dict_df['{}'.format(aspect[a])].tolist()

# Similar aspect
authenticity = seed_dict_df['authenticity'].tolist()
delivery = seed_dict_df['delivery'].tolist()
price = seed_dict_df['price'].tolist()
service = seed_dict_df['service'].tolist()

authenticity = [x for x in authenticity if x in list(word2id.keys())]
delivery = [x for x in delivery if x in list(word2id.keys())]
price = [x for x in price if x in list(word2id.keys())]
service = [x for x in service if x in list(word2id.keys())]

print('authenticity: {}'.format(authenticity))
print('delivery: {}'.format(delivery))
print('price: {}'.format(price))
print('service: {}'.format(service))

# Mebe only aspect
if domain == 'mebe':
    quality = seed_dict_df['quality'].tolist()
    safety = seed_dict_df['safety'].tolist()

    quality = [x for x in quality if x in list(word2id.keys())]
    safety = [x for x in safety if x in list(word2id.keys())]

    print('quality: {}'.format(quality))
    print('safety: {}'.format(safety))

# Tech only aspect
if domain == 'tech':
    appearance = seed_dict_df['appearance'].tolist()
    accessories = seed_dict_df['accessories'].tolist()
    hardware = seed_dict_df['hardware'].tolist()
    performance = seed_dict_df['performance'].tolist()

    appearance = [x for x in appearance if x in list(word2id.keys())]
    accessories = [x for x in accessories if x in list(word2id.keys())]
    hardware = [x for x in hardware if x in list(word2id.keys())]
    performance = [x for x in performance if x in list(word2id.keys())]

    print('appearance: {}'.format(appearance))
    print('accessories: {}'.format(accessories))
    print('hardware: {}'.format(hardware))
    print('performance: {}'.format(performance))

model = guidedlda.GuidedLDA(n_topics=len(aspect), n_iter=1000, random_state=15, refresh=20, alpha=1, eta=0.01)
seed_topic_list = aspect
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        try:
            seed_topics[word2id[word]] = t_id
        except KeyError:
            continue

model.fit(X.toarray().astype(int), seed_topics=seed_topics, seed_confidence=1.1-seed_round/10)
# joblib.dump(model, "Guided_LDA_6topics.pkl")
n_top_words = seed_round*30
topic_word = model.topic_word_
next_seed_df = pd.DataFrame()
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('{}: {}'.format(i, ' '.join(topic_words)))

    # df = pd.DataFrame(topic_words)
    # # df.to_csv(os.path.join(dirname, 'dict/{}'.format(l) + '{}'.format(name + str(f.split('/')[1]))), ',', encoding='utf-8')
    # next_seed_df = pd.concat([next_seed_df, df], axis=1)

