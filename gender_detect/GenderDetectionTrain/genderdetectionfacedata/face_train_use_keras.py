#-*- coding: utf-8 -*-
import random

import numpy as np
#from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

from load_face_dataset import resize_image, IMAGE_SIZE

# CNN网络模型类
class Model:
    def __init__(self):
        self.model = None

    def load_model(self, file_path):
        self.model = load_model(file_path)

    # 识别
    def face_predict(self, image):
        image = resize_image(image)
        image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))

        # 浮点并归一化
        image = image.astype('float32')
        image /= 255

        # 给出输入属于各个类别的概率，我们是二值类别，则该函数会给出输入图像属于0和1的概率各为多少
        result = self.model.predict(image)
        #print('result:', result)
        gender_class = np.argmax(result)
        print(gender_class)

        # 返回类别预测结果
        return gender_class

