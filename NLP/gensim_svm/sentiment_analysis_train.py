#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 21:38:33 2019

@author: luke
"""

# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec #gensim-1.0.0
import numpy as np
import pandas as pd
import jieba
from sklearn.externals import joblib
from sklearn.svm import SVC

def load_file_and_preprocessing():
    neg=pd.read_excel('data/neg.xls',header=None,index=None)
    pos=pd.read_excel('data/pos.xls',header=None,index=None)

    #分词
    cw = lambda x: list(jieba.cut(x))
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    print type(pos['words'])
    print(pos['words'])
    quit()
    
    #use 1 for positive sentiment, 0 for negative
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))))
    x = np.concatenate((pos['words'], neg['words']))
    print(type(x))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #print(x_train)
    np.save('svm_data/y_train.npy',y_train)
    np.save('svm_data/y_test.npy',y_test)
    return x_train,x_test

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

#计算词向量
def get_train_vecs(x_train,x_test):
    n_dim = 300
    #初始化模型和词表
    imdb_w2v = Word2Vec(size=n_dim, min_count=10)
    imdb_w2v.build_vocab(x_train)

    #在评论训练集上建模(可能会花费几分钟)
    imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)
    #print(type(imdb_w2v))

    train_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_train])
    #train_vecs = scale(train_vecs)

    np.save('svm_data/train_vecs.npy',train_vecs)
    print train_vecs.shape
    #在测试集上训练
    imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)
    imdb_w2v.save('svm_data/w2v_model/w2v_model.pkl')
    #Build test tweet vectors then scale
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim,imdb_w2v) for z in x_test])
    #test_vecs = scale(test_vecs)
    np.save('svm_data/test_vecs.npy',test_vecs)
    print test_vecs.shape

def get_data():
    train_vecs=np.load('svm_data/train_vecs.npy')
    y_train=np.load('svm_data/y_train.npy')
    test_vecs=np.load('svm_data/test_vecs.npy')
    y_test=np.load('svm_data/y_test.npy') 
    return train_vecs,y_train,test_vecs,y_test

def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf=SVC()
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'svm_data/svm_model/model.pkl')
    print clf.score(test_vecs,y_test)

x_train,x_test=load_file_and_preprocessing()
get_train_vecs(x_train,x_test)
train_vecs,y_train,test_vecs,y_test=get_data()

svm_train(train_vecs,y_train,test_vecs,y_test)
