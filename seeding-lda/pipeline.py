import datetime
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from load_data import load_polarity_data

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modules.polarity.model import SentimentModel

TRAIN_PATH = 'data/input/SP/tech_shopee_train.csv'
TEST_PATH = 'data/input/SP/tech_shopee_test.csv'


def train_SP_chi2(aspect, seed, X_train, y_train, X_test, y_test, model, save=True):
    print('=' * 20 + ' Training SP with Chi2 ' + aspect + ' ' + (50 - len(' Training SP with Chi2 ')) * '=')
    model.chi2_dict_lda()
    model.lda_dict()
    _y_train = [i.score for i in y_train]
    _y_test = [i.score for i in y_test]
    print('* Representing using Chi2 ...')
    _X_train = model.chi2_represent(X_train)
    _X_test = model.chi2_represent(X_test)
    print('  Representing using Chi2 DONE!')

    print('* Training ...')
    model.train(_X_train, _y_train)
    print('  Training DONE!')
    predict = model.predict(_X_test)

    neg_p, neg_r, neg_f1 = model.evaluate_lda(_y_test, predict, -1)
    pos_p, pos_r, pos_f1 = model.evaluate_lda(_y_test, predict, 1)
    print("- p negative     :", neg_p)
    print("- r negative     :", neg_r)
    print("- F1 negative      :", neg_f1)
    print("- p positive     :", pos_p)
    print("- r positive     :", pos_r)
    print("- F1 positive      :", pos_f1)

    result = pd.DataFrame({'score': [aspect, seed, model.model, neg_p, neg_r, neg_f1, neg_p, neg_r, neg_f1]},
                          index=['Aspect', 'seed', 'Model', 'p-negative', 'r-negative', 'F1-negative', 'p-positive',
                                 'r-positive', 'F1-positive'])

    if save:
        result.to_csv('./data/output/evaluate/SP/lda/SP_result_{m}_{a}_seed{seed}.csv'.format(
            m=model.model,
            a=aspect,
            seed=seed
        ))
    # return neg_f1, pos_f1
    return neg_p, neg_r, neg_f1, pos_p, pos_r, pos_f1


if __name__ == '__main__':
    test_df = pd.read_csv(TEST_PATH)
    print(test_df.giá.value_counts())
    # model = SVC()
    # model = DecisionTreeClassifier()
    # model = KNeighborsClassifier()
    model = LogisticRegression()
    # model = MultinomialNB()
    # model = RandomForestClassifier()

    aspect_list = ['giá', 'dịch_vụ', 'ship', 'hiệu_năng', 'chính_hãng', 'cấu_hình', 'phụ_kiện', 'mẫu_mã']
    neg_p = []
    neg_r = []
    neg_f1 = []
    pos_p = []
    pos_r = []
    pos_f1 = []
    seed = 6
    domain = 'tech_shopee'
    DOMAIN = 'tech'
    for aspect in aspect_list:
        X_train, y_train = load_polarity_data(path=TRAIN_PATH,
                                              stc_idx_col_name='id',
                                              stc_col_name='cmt',
                                              label_col_name=aspect)

        X_test, y_test = load_polarity_data(path=TEST_PATH,
                                            stc_idx_col_name='id',
                                            stc_col_name='cmt',
                                            label_col_name=aspect)
        vocab_path_chi2 = './data/output/chi2_score_dict/SP_{domain}_{aspect}.csv'.format(domain=domain,aspect=aspect)
        vocab_path_lda = './data/output/lda/total_seed{seed}/{domain}_{aspect}_seed{seed}.csv'.format(domain=DOMAIN,
                                                                                                  seed=seed,
                                                                                                  aspect=aspect)
        SP_model = SentimentModel(vocab_path_chi2, vocab_path_lda, model)
        # _neg_f1, _pos_f1 = train_SP_chi2(aspect, seed, X_train, y_train, X_test, y_test, SP_model, save=False)
        # neg_f1.append(_neg_f1)
        # pos_f1.append(_pos_f1)
        _neg_p, _neg_r, _neg_f1, _pos_p, _pos_r, _pos_f1 = train_SP_chi2(aspect, seed, X_train, y_train, X_test, y_test, SP_model, save=False)
        neg_p.append(_neg_p)
        neg_r.append(_neg_r)
        neg_f1.append(_neg_f1)
        pos_p.append(_pos_p)
        pos_r.append(_pos_r)
        pos_f1.append(_pos_f1)
    macro_neg_p = np.array(neg_p).mean()
    macro_pos_p = np.array(pos_p).mean()
    macro_neg_r = np.array(neg_r).mean()
    macro_pos_r = np.array(pos_r).mean()
    macro_neg_f1 = np.array(neg_f1).mean()
    macro_pos_f1 = np.array(pos_f1).mean()
    print('=' * 20 + ' Performance of SP model ' + (50 - len(' Performance of SP model ')) * '=')

    print("- Macro-P positive  ", macro_pos_p)
    print("- Macro-R positive  ", macro_pos_r)
    print("- Macro-F1 positive ", macro_pos_f1)
    print("- Macro-P negative  ", macro_neg_p)
    print("- Macro-R negative  ", macro_neg_r)
    print("- Macro-F1 negative ", macro_neg_f1)

    result = pd.DataFrame({'score': [model, macro_pos_p, macro_pos_r, macro_pos_f1, macro_neg_p, macro_neg_r, macro_neg_f1]},
                          index=['Model', 'Macro-P positive', 'Macro-R positive', 'Macro-F1 positive', 'Macro-P negative', 'Macro-R negative', 'Macro-F1 negative'])
    result.to_csv(
        './data/output/evaluate/SP/lda/SP_result_{m}_seed{seed}.csv'.format(m=model, seed=seed))
