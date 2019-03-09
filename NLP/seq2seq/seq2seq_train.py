#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 11:30:25 2019

@author: luke
"""

from keras.layers import Input,LSTM,Dense
from keras.models import Model,load_model
from keras.callbacks import ModelCheckpoint

import pandas as pd
import numpy as np

import pickle

def create_model(n_input,n_output,n_units):
    #训练阶段
    #encoder
    encoder_input = Input(shape = (None, n_input))
    #encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
    encoder = LSTM(n_units, return_state=True)
    #n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
    _,encoder_h,encoder_c = encoder(encoder_input)
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

input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

INUPT_LENGTH = max([len(i) for i in input_texts])
OUTPUT_LENGTH = max([len(i) for i in target_texts])
INPUT_FEATURE_LENGTH = len(input_characters)
OUTPUT_FEATURE_LENGTH = len(target_characters)

encoder_input = np.zeros((NUM_SAMPLES,INUPT_LENGTH,INPUT_FEATURE_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))

input_dict = {char:index for index,char in enumerate(input_characters)}
print(input_dict)
input_dict_reverse = {index:char for index,char in enumerate(input_characters)}
target_dict = {char:index for index,char in enumerate(target_characters)}
print(target_dict)
target_dict_reverse = {index:char for index,char in enumerate(target_characters)}

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

for seq_index,seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index,char_index,input_dict[char]] = 1

for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0

#print(''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) !=0]))

#print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) !=0]))

#print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) !=0]))

model_train, encoder_infer, decoder_infer = \
    create_model(INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)

model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model_train.summary()

encoder_infer.summary()

decoder_infer.summary()

model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=BATCH_SIZE,
                epochs=EPOCH,validation_split=0.2)
model_train.save("translate.h5")
encoder_infer.save("encoder_infer.h5")
decoder_infer.save("decoder_infer.h5")
