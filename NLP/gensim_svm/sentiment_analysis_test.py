#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:36:43 2019

@author: luke
"""

import jieba
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.externals import joblib

def build_sentence_vector(text, size,imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load('svm_data/w2v_model/w2v_model.pkl')
    #imdb_w2v.train(words)
    train_vecs = build_sentence_vector(words, n_dim,imdb_w2v)
    #print train_vecs.shape
    return train_vecs

def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('svm_data/svm_model/model.pkl')
    print(type(clf))

    result=clf.predict(words_vecs)

    return result

##对输入句子情感进行判断
#string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
string='还行'
print(svm_predict(string))
