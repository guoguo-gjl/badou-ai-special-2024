import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
# 【placeholder是什么？】
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])
# 定义中间层
weights_l1 = tf.Variable(tf.random_normal([1, 10]))
biases_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1
l1 = tf.nn.tanh(wx_plus_b_l1)  # 加入激活函数
# 定义输出层
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
biases_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(l1, weights_l2) + biases_l2
predicition = tf.nn.tanh(wx_plus_b_l2)
# 定义损失函数
loss = tf.reduce_mean(tf.square(y-predicition))  # y-prediction
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})
        predicition_value = sess.run(predicition, feed_dict={x:x_data})
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, predicition_value, 'r', lw=5)
        plt.show()
