from __future__ import absolute_import
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda
from keras import backend as K
import cv2
import numpy as np

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

img1 = cv2.imread("mnist_data/6/4.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("mnist_data/6/9.jpg", cv2.IMREAD_GRAYSCALE)

img1 = img1.astype('float32')
img2 = img2.astype('float32')

img1 /= 255
img2 /= 255

# network definition
base_network = create_base_network(img1.shape)

input_a = Input(shape=img1.shape)
input_b = Input(shape=img1.shape)

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance)([processed_a, processed_b])

model = Model([input_a, input_b], distance)

model.load_weights('my_model.h5')

distance1 = model.predict([np.array([img1]), np.array([img2])])
print(distance1)

img3 = cv2.imread("mnist_data/7/5.jpg", cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread("mnist_data/3/5.jpg", cv2.IMREAD_GRAYSCALE)

img3 = img3.astype('float32')
img4 = img4.astype('float32')

img3 /= 255
img4 /= 255


distance2 = model.predict([np.array([img3]), np.array([img4])])
print(distance2)
