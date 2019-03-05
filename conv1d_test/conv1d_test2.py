import tensorflow as tf
import numpy as np

a = tf.constant([1,2,3,4,5],dtype=np.float16)
a = tf.reshape(a,[-1,5,1])
b = tf.constant([1,2],dtype=np.float16)
b = tf.reshape(b,[2,1,1])
 
c = tf.nn.conv1d(a,b,stride=2,padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(c))
