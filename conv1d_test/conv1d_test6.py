import tensorflow as tf
import numpy as np

# 定义一个矩阵a，表示需要被卷积的矩阵。
a = np.ones([1, 5, 20], dtype=np.float32)


# 卷积核，此处卷积核的数目为1
kernel = np.array(np.arange(1, 1 + 40), dtype=np.float32).reshape([2, 20, 1])

print(a)
print(a.shape)
print(kernel)
print(kernel.shape)

# 进行conv1d卷积
conv1d = tf.nn.conv1d(a, kernel, 1, 'SAME')

with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run(conv1d))
