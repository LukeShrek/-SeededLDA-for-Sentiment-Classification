import datetime

import pandas as pd
from typing import List, Tuple
from vncorenlp import VnCoreNLP
import numpy as np

from models import NotTagNerYetInput, NerInput, NerOutput
from modules.models import PivyTokenizer
from modules.ner.preprocess import export_to_file
from pyvi import ViTokenizer, ViPosTagger

def preprocess(inputs):
    for input in inputs:
        input.stc = tokenize(input.stc)
        input.stc = stc_to_word_list(input.stc)

    return inputs


def load_stopword(path):
    """
    load stopword from path,
    :param path:
    :return: pd.DataFrame
    """
    return pd.read_csv(path, sep=',', header=None, names=["stopword"])


def tokenize(stc: str) -> np.array:
    tokenizer = PivyTokenizer()
    tokenize_input = tokenizer.tokenize(stc)
    return tokenize_input


def pos_tag(stc: str) -> np.array:
    tokenizer = PivyTokenizer()
    pos_tag_list = tokenizer.pos_tag(stc)
    return pos_tag_list




def stc_to_word_list(input: str) -> list:
    input = input.split(' ')
    return input


IBO_ENCODE_DICT = {'O': 0, 'B-SUB': 1, 'I-SUB': 2, 'B-OBJ': 3, 'I-OBJ': 4}


def tag_IBO(inputs: List[NotTagNerYetInput], save=True) -> Tuple[List[NerInput], List[NerOutput]]:
    tokenize_word_list = []
    IBO_tags = []
    tokenize_stcs = []
    IBO_stc_tags = []
    stc_idxs = []
    pos_tags = []
    word_id = 0
    for input in inputs:
        input.stc = tokenize(input.stc)
        input.pos = pos_tag(input.stc)

        input.subject = tokenize(input.subject)
        input.object = tokenize(input.object)
        stc_idxs.append(input.stc_idx)
        stc_contain_IB = input.stc
        # print(input.object)
        if input.subject != 'undefined':
            stc_contain_IB = input.stc.replace(input.subject, 'B-SUB' + (' I-SUB' * (len(input.subject.split(' ')) - 1)))
            if 'B-S' not in stc_contain_IB:
                S_list = input.subject.split('_')
                S_true_list = []
                for i in S_list:
                    S_true_list = S_true_list + i.split(' ')
                stc_contain_IB = stc_contain_IB.replace(S_true_list[0], 'B-SUB')
                for i in S_true_list[1:]:
                    stc_contain_IB = stc_contain_IB.replace(i, 'I-SUB')

        if input.object != 'undefined':
            stc_contain_IB = stc_contain_IB.replace(input.object, 'B-OBJ' + (' I-OBJ' * (len(input.object.split(' ')) - 1)))
            if 'B-O' not in stc_contain_IB:
                O_list = input.object.split('_')
                O_true_list = []
                for i in O_list:
                    O_true_list = O_true_list + i.split(' ')
                print('O', O_true_list)
                stc_contain_IB = stc_contain_IB.replace(O_true_list[0], 'B-OBJ')
                for i in O_true_list[1:]:
                    stc_contain_IB = stc_contain_IB.replace(i, 'I-OBJ')

        input.stc = stc_to_word_list(input.stc)
        tokenize_stcs.append(input.stc)
        stc_contain_IB_list = stc_to_word_list(stc_contain_IB)
        stc_IBO_list = ['O' if word not in ['B-SUB', 'I-SUB', 'B-OBJ', 'I-OBJ'] else word for word in stc_contain_IB_list]
        IBO_stc_tags.append([IBO_ENCODE_DICT[i] for i in stc_IBO_list])
        # pos_tags.append(input.pos)
        # render các từ trong toàn bộ corpus thành mảng
        for i in range(0, len(input.stc)):
            print('x', input.stc)
            print('y', stc_IBO_list)
            tokenize_word_list.append(NerInput(word_id, input.stc[i]))
            pos_tags.append(input.pos[i])
            IBO_tags.append(NerOutput(stc_IBO_list[i]))

        word_id = word_id+1
    if save == True:
        export_to_file('./data/output/IBO_tag_data/IBO_tag_stc_data_{}.txt'.format(
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), tokenize_stcs, IBO_stc_tags, stc_idxs)
        # IBO_tags_stc_df = pd.DataFrame({'word_idx': stc_idxs,
        #                                 'stc': tokenize_stcs,
        #                                 'IBO_tag': IBO_stc_tags})
        IBO_tag_data_df = pd.DataFrame({'Word_idx': [w.word_idx for w in tokenize_word_list],
                                        'Word': [w.word for w in tokenize_word_list],
                                        'POS': [p for p in pos_tags],
                                        'Tag': [tag.score for tag in IBO_tags],
                                        })
        # IBO_tags_stc_df.to_csv(
        #     './data/output/IBO_tag_data/IBO_tag_stc_data_{}.csv'.format(
        #         datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
        #     index=False)
        IBO_tag_data_df.to_csv(
            './data/output/IBO_tag_data/IBO_tag_data_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')),
            index=False)

    return tokenize_word_list, IBO_tags
