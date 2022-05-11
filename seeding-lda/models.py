# class Document:
#
# class Sentence:
#
# class Token:
from typing import Any


class Input:
    def __init__(self, stc_idx: int, stc: str) -> None:
        self.stc_idx = stc_idx
        self.stc = stc


class ComparativeStcOutput:
    def __init__(self, score: int) -> None:
        self.score = score


class AspectOutput:
    def __init__(self, aspects: list, scores: list) -> None:
        self.aspects = aspects
        self.scores = scores


class PolarityOutput:
    def __init__(self, aspects: list, scores: list) -> None:
        self.aspects = aspects
        self.scores = scores


class NotTagNerYetInput:
    def __init__(self, stc_idx: int, stc: str, subject: str, object: str) -> None:
        self.subject = subject
        self.object = object
        self.stc_idx = stc_idx
        self.stc = stc


class NerInput:
    def __init__(self, word_idx, word):
        self.word_idx = word_idx
        self.word = word


class NerOutput:
    def __init__(self, score):
        self.score = score
