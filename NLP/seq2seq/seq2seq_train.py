#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:30:25 2019

@author: luke
"""

from keras.layers import Input,LSTM,Dense,Embedding
from keras.models import Model
#from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np
import re

import pickle

EMBED_HIDDEN_SIZE = 50

def create_model(input_length, n_input,n_output,n_units):
    #训练阶段
    #encoder
    encoder_input = Input(shape = (input_length,))
    embedded_input = Embedding(n_input, EMBED_HIDDEN_SIZE)(encoder_input)
    #encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    #n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _,encoder_h,encoder_c = encoder(embedded_input)
    encoder_state = [encoder_h,encoder_c]
    #保留下来encoder的末状态作为decoder的初始状态

    #decoder
    decoder_input = Input(shape = (None, n_output))
    #decoder的输入维度为中文字符数
    decoder = LSTM(n_units,return_sequences=True, return_state=True)
    #训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
    decoder_output, _, _ = decoder(decoder_input,initial_state=encoder_state)
    #在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
    decoder_dense = Dense(n_output,activation='softmax')
    decoder_output = decoder_dense(decoder_output)
    #输出序列经过全连接层得到结果

    #生成的训练模型
    model = Model([encoder_input,decoder_input],decoder_output)
    #第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出

    #推理阶段，用于预测过程
    #推断模型—encoder
    encoder_infer = Model(encoder_input,encoder_state)

    #推断模型-decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]#上个时刻的状态h,c

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]#当前时刻得到的状态
    decoder_infer_output = decoder_dense(decoder_infer_output)#当前时刻的输出
    decoder_infer = Model([decoder_input]+decoder_state_input,[decoder_infer_output]+decoder_infer_state)

    return model, encoder_infer, decoder_infer

def build_covab(articles):
    d = {}  # {'word' : num}
    for article in articles:
        for word in article:
            if word in d:
                d[word] += 1
            else:
                d[word] = 1
    s = sorted(d.items(), key=lambda x:x[1], reverse=True) #降序

#   计算最佳词典长度
#    s_n = [x[1] for x in s]
#    s_n = s_n[1:]  #去掉<pad>
#    s_n = np.asarray(s_n)
#    print("s_n.mean",s_n.mean())
#    print(s_n[int(len(s_n)*0.3)])  #取出所有词中频率最高的词的前30%，DEBUG_MODE中出现了2次
#    print(int(len(s_n)*0.3))  #9066  #VOCAB_SIZE = 9000

    s_w = [x[0] for x in s]
    word2int = {v : k + 1 for k, v in enumerate(s_w)}
    int2word = {k + 1 : v for k, v in enumerate(s_w)}
    return word2int, int2word

def clean_str_to_list(string):
    #string = re.sub(r"[^A-Za-z0-9(),!?\']", " ", string)
    string = re.sub(r"n\'t", " n\'", string) #don't -> do n't(do not)
    string = re.sub(r"\'s", " \'s", string) #It's -> It 's(It is or It has)
    string = re.sub(r"\'ve", " \'ve", string) #I've -> I 've(I have)
    string = re.sub(r"\'re", " \'re", string) #You're -> You 're(You are)
    string = re.sub(r"\'d", " \'d", string) #I'd (like to) -> I 'd(I had)
    string = re.sub(r"\'ll", " \'ll", string) #I'll -> I 'll(I will)
    string = re.sub(r"\.", " . ", string) #',' -> ' , '
    string = re.sub(r",", " , ", string) #',' -> ' , '
    string = re.sub(r"!", " ! ", string) #'!' -> ' ! '
    string = re.sub(r"\(", " ( ", string) #'(' -> ' ( '
    string = re.sub(r"\)", " ) ", string) #')' -> ' ) '
    string = re.sub(r"\?", " ? ", string) #'?' -> ' ? '
    sentense=[]
    for word in string.split(" "):
        if word.strip():
             sentense.append(word)
    return sentense

N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 100
NUM_SAMPLES = 10000

data_path = 'data/cmn.txt'

df = pd.read_table(data_path,header=None).iloc[:NUM_SAMPLES,:,]
df.columns=['inputs','targets']

df['targets'] = df['targets'].apply(lambda x: '\t'+x+'\n')

input_texts = df.inputs.values.tolist()
target_texts = df.targets.values.tolist()

input_words = []
for i in range(len(input_texts)):
    input_words.append(clean_str_to_list(input_texts[i]))

target_texts = [i.strip() for i in target_texts]
#print(input_texts)
#print(target_texts)

#print(input_words)

#input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

INUPT_LENGTH = max([len(i) for i in input_words])
OUTPUT_LENGTH = max([len(i) for i in target_texts])

OUTPUT_FEATURE_LENGTH = len(target_characters)

encoder_input = np.zeros((NUM_SAMPLES,INUPT_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))

#input_dict = dict((char,index + 1) for index,char in enumerate(input_characters))
#print(input_dict)
#input_dict_reverse = dict((index + 1,char) for index,char in enumerate(input_characters))
input_dict, input_dict_reverse = build_covab(input_words)
#print(input_dict[:10])
#print(input_dict_reverse[:10])

target_dict = {char:index for index,char in enumerate(target_characters)}
#print(target_dict)
target_dict_reverse = {index:char for index,char in enumerate(target_characters)}

INPUT_FEATURE_LENGTH = len(input_dict) + 1

print("OUTPUT_LENGTH=" , OUTPUT_LENGTH)
print("OUTPUT_FEATURE_LENGTH=" , OUTPUT_FEATURE_LENGTH)
print(len(input_dict))
print(len(target_dict))

with open("output_length.pkl", "wb") as f:
    pickle.dump([INUPT_LENGTH, OUTPUT_LENGTH], f)

with open("input_dict.pkl", "wb") as f:
    pickle.dump(input_dict, f)

with open("input_dict_reverse.pkl", "wb") as f:
    pickle.dump(input_dict_reverse, f)

with open("target_dict.pkl", "wb") as f:
    pickle.dump(target_dict, f)

with open("target_dict_reverse.pkl", "wb") as f:
    pickle.dump(target_dict_reverse, f)

for seq_index,seq in enumerate(input_words):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index,char_index] = input_dict[char]

for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0

#print(''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) !=0]))

#print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) !=0]))

#print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) !=0]))

model_train, encoder_infer, decoder_infer = \
    create_model(INUPT_LENGTH, INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)

model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model_train.summary()

encoder_infer.summary()

decoder_infer.summary()

model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,
                epochs=EPOCH,validation_split=0.2)
model_train.save("translate.h5")
encoder_infer.save("encoder_infer.h5")
decoder_infer.save("decoder_infer.h5")
