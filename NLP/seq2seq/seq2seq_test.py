#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 11:17:16 2019

@author: luke
"""

from keras.models import load_model
import numpy as np

import pickle
import re


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

input_dict = pickle.load(open('input_dict.pkl', 'rb'))
target_dict = pickle.load(open('target_dict.pkl', 'rb'))
target_dict_reverse = pickle.load(open('target_dict_reverse.pkl', 'rb'))

INUPT_LENGTH, OUTPUT_LENGTH = pickle.load(open('output_length.pkl', 'rb'))
print(INUPT_LENGTH)
print(OUTPUT_LENGTH)
OUTPUT_FEATURE_LENGTH = len(target_dict)
print(OUTPUT_FEATURE_LENGTH)
INPUT_FEATURE_LENGTH = len(input_dict)
print(INPUT_FEATURE_LENGTH)

print(input_dict)

encoder_input = np.zeros((1,INUPT_LENGTH))


encoder_infer = load_model("encoder_infer.h5")
decoder_infer = load_model("decoder_infer.h5")


def predict_chinese(source, encoder_inference, decoder_inference, n_steps, features):
    source = clean_str_to_list(source)
    for char_index, char in enumerate(source):
        encoder_input[0,char_index] = input_dict[char]

    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(encoder_input)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1

    print(state)
    print(type(state))
    print(len(state))

    print(predict_seq)
    print([predict_seq])
    print(type(predict_seq))
    print(type([predict_seq]))

    c = [predict_seq]+state
    print(c)
    print(type(c))
    print(len(c))

    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':#预测到了终止符则停下来
            break
    return output

input_str = "Do you remember?"
out = predict_chinese(input_str, encoder_infer, decoder_infer, OUTPUT_LENGTH,
                      OUTPUT_FEATURE_LENGTH)
#print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
print(input_str)
print(out)
