import pandas as pd
from models import Input, PolarityOutput, ComparativeStcOutput, AspectOutput, NotTagNerYetInput
from models import NerInput, NerOutput


def load_comparative_stc_data(
        path: str,
        stc_idx_col_name: str,
        stc_col_name: str,
        label_col_name: str) -> object:
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        stc_idx = row[stc_idx_col_name]
        stc = row[stc_col_name].strip()
        inputs.append(Input(stc_idx, stc))
        outputs.append(ComparativeStcOutput(row[label_col_name]))

    return inputs, outputs


def load_aspect_data(
        path: str,
        stc_idx_col_name: str,
        stc_col_name: str,
        aspect_col_names: list) -> object:
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        stc_idx = row[stc_idx_col_name]
        stc = row[stc_col_name].strip()
        inputs.append(Input(stc_idx, stc))
        aspects = aspect_col_names
        scores = list(row(aspect_col_names))
        outputs.append(AspectOutput(aspects, scores))

    return inputs, outputs


def load_polarity_data(
        path: str,
        stc_idx_col_name: str,
        stc_col_name: str,
        label_col_name: str) -> object:
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        try:
            if int(row[label_col_name]) != 0:
                # print('a', row[label_col_name])
                stc_idx = row[stc_idx_col_name]
                stc = row[stc_col_name].strip()
                inputs.append(Input(stc_idx, stc))
                outputs.append(ComparativeStcOutput(int(row[label_col_name])))
        except: pass
    return inputs, outputs


def load_not_tag_ner_yet_data(
        df: pd.DataFrame,
        stc_idx_col_name: str,
        stc_col_name: str,
        subject_col_name: str,
        object_col_name: str) -> list:
    inputs = []
    df = df.fillna('undefined')
    for _, row in df.iterrows():
        stc_idx = row[stc_idx_col_name]
        stc = row[stc_col_name]
        subject = row[subject_col_name].strip()
        object = row[object_col_name].strip()
        inputs.append(NotTagNerYetInput(stc_idx, stc, subject, object))

    return inputs


def load_ner_data(
        path: object,
        word_idx_col_name: str,
        word_col_name: str,
        ner_tag_col_name: str) -> object:
    inputs = []
    outputs = []
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        word_idx = row[word_idx_col_name]
        word = row[word_col_name]
        inputs.append(NerInput(word_idx, word))
        outputs.append(NerOutput(row[ner_tag_col_name]))

    return inputs, outputs
