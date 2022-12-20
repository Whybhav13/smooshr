import streamlit as st
import numpy as np
import pandas as pd
import os
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import math
import sys
import json
import ast

# to avoid typing st.session_state multiple times
state = st.session_state


def repeat(x):
    _size = len(x)
    repeated = []
    for _i in range(_size):
        k = _i + 1
        for _j in range(k, _size):
            if x[_i] == x[_j] and x[_i] not in repeated:
                repeated.append(x[_i])
    return repeated


def new_columns(_result):
    _columns = []
    for _, _rows in _result.iterrows():

        if _rows[2] > 0.98:
            _columns.append(_rows[0])
        else:
            _columns.append(_rows[1])

    print(repeat(_columns))
    return _columns


def ngrams(string, n=3):
    string = re.sub(r'[,-./]|\sBD', r'', string)
    _ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in _ngrams]


def _column_maker(_archive, _new):
    # company_names = new[0].columns.values.tolist()
    # company_match = new[-2].columns.values.tolist()
    if _archive is None:
        _archive = _new
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    temp = _new.copy()
    temp.extend(_archive)
    vectorizer.fit(temp)
    tf_idf_matrix = vectorizer.transform(_archive)
    tf_idf_matrix_match = vectorizer.transform(_new)

    answer = cosine_similarity(tf_idf_matrix, tf_idf_matrix_match)
    idx = answer.argmax(axis=0)
    value = answer.max(axis=0)

    ans = []
    j = 0
    for i in idx:
        temp = [_archive[i], _new[j], value[j]]
        j += 1
        ans.append(temp)

    result = pd.DataFrame(ans)
    result = result.sort_values(2, ascending=False).reset_index(drop=True)

    return result


def user_prompt(_new):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
    tf_idf_matrix = vectorizer.fit_transform(_new)

    answer = cosine_similarity(tf_idf_matrix, tf_idf_matrix)

    idx = answer.argsort(axis=0)[-2]
    t = answer.copy()
    t.sort(axis=0)
    value = t[-2]

    ans = []
    j = 0
    for i in idx:
        temp = [_new[i], _new[j], value[j]]
        j += 1
        ans.append(temp)

    result = pd.DataFrame(ans)
    result = result.sort_values(0)
    voila = []
    score = []

    for _, r in result.iterrows():
        temp = r[:2].tolist()
        _temp = temp.copy()
        _temp.reverse()
        if _temp not in voila:
            voila.append(temp)
            score.append(r[2])

    df = pd.DataFrame(voila)
    df[2] = score
    return df.sort_values(2, ascending=False)


def attach(series1, series2):
    ret = []
    for s1, s2 in zip(series1, series2):
        if ~math.isnan(s1) and ~math.isnan(s2):
            if math.isnan(s1):
                ret.append(s2)
            else:
                ret.append(s1)
        else:
            print("oops")
    return ret


def final_df(_config, _final):
    for c in _config:
        _final[c[0]] = attach(_final[c[0]], _final[c[1]])
        _final.drop(c[1], axis=1, inplace=True)
    return _final


def transform(_base):
    print(_base)

    files = os.listdir(_base)
    args = pd.to_datetime([name.replace("1-", "1-1-").replace(".xls", "") for name in files]).argsort()
    files = [files[arg] for arg in args]

    new = []
    names = []
    for name in files:

        path = os.path.join(_base, name)
        df = pd.read_excel(path)

        # print(name)

        check = df.columns.astype(str).str.contains("Unnamed").sum()
        if check:
            temp = df.dropna(thresh=df.shape[1] // 1.5).dropna(axis=1, thresh=df.shape[0] // 4)
            header = temp.iloc[0]
            temp.columns = header
            temp = temp[1:]
            nulls = temp.apply(lambda s: pd.to_numeric(s, errors='coerce').isna()).sum().sum()
            total = temp.shape[0] * temp.shape[1]
            if nulls / total < 0.7:
                new.append(temp)
                names.append(name.split(".")[0])
        else:

            nulls = df.apply(lambda s: pd.to_numeric(s, errors='coerce').isna()).sum().sum()
            total = df.shape[0] * df.shape[1]
            if nulls / total < 0.7:
                new.append(df)
                names.append(name.split(".")[0])

    return pd.concat(new, keys=names)


base = st.text_input("hello")

final = transform(base)
disp = user_prompt(final.columns)

st.title("Which of these are good recs:")
with st.form("key"):
    for _, data in disp[disp[2] > 0.5].iterrows():
        st.checkbox(data[0], key="1" + str(_))
        st.checkbox(data[1], key="2" + str(_))
        st.checkbox("rec3", key="3" + str(_), value=True)
        st.text("---" * 40)

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    print("yaya")
    config = []
    for _, data in disp[disp[2] > 0.5].iterrows():
        a, b, c = state["1" + str(_)], state["2" + str(_)], state["3" + str(_)]
        if not c:
            if a:
                config.append([data[0], data[1]])
            else:
                config.append([data[1], data[0]])

    final = final_df(config, final)
    print(config)
    st.write(final)
    print(final.shape)
