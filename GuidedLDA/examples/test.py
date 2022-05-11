# Importing modules

import pandas as pd
import numpy as np
import time
import re
import os
from os import path
from pprint import pprint
import joblib
import sys
sys.path.insert(1, './scripts/')
# from nlpuitls import NLP

#NLP
import sklearn
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import guidedlda

# #plotting
# import matplotlib.pyplot as plt
# import pyLDAvis
# import pyLDAvis.sklearn
# pyLDAvis.enable_notebook()
# import matplotlib.pyplot as plt
# %matplotlib inline

file_path = path.relpath("./Opinion-Mining/data/raw_data/tech_tiki.csv")
with open(file_path) as f:
    df = pd.read_csv(f)
    df.head()