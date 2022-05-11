from abc import ABC

import numpy as np


class Model:
    def train(self, inputs, outputs):
        """

        :param inputs:
        :param outputs:
        :return:
        """
        raise NotImplementedError

    def save(self):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def load(self, path):
        """

        :param path:
        :return:
        """
        raise NotImplementedError

    def predict(self, inputs):
        """

        :param inputs: list of Input object
        :return:
        :rtype: list of models.evaluate
        """
        raise NotImplementedError

    # def evaluate(self, y_test, y_predicts):
    #     raise NotImplementedError


class Tokenizer():
    def tokenize(self, inputs):
        """

        :param inputs:
        """
        raise NotImplementedError

    def pos_tag(self, inputs):
        """
        :param inputs:
        """
        raise NotImplementedError

    def chunk_tag(self, inputs):
        raise NotImplementedError


class VnCoreNLPTokenizer(Tokenizer):
    def __init__(self):
        self.model = VnCoreNLP("vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg,pos,parse",
                               max_heap_size='-Xmx2g')

    def tokenize(self, stc):
        tokenize_input = self.model.tokenize(stc)
        stc = ' '.join(np.array(tokenize_input).ravel())
        return stc

    def pos_tag(self, inputs):
        """

        :param inputs:
        """
        pass

    def chunk_tag(self, inputs):
        pass


class PivyTokenizer(Tokenizer, ABC):
    def __init__(self):
        pass

    def tokenize(self, stc):
        stc = ViTokenizer.tokenize(stc)
        return stc

    def pos_tag(self, stc):
        """

        :param inputs: stc has been tokenize and joined to string
        """

        pos_tags = ViPosTagger.postagging(stc)[1]
        # print(pos_tags)
        return pos_tags

    def chunk_tag(self, inputs):
        pass
