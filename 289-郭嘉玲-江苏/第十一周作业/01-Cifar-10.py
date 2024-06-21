import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

# 1.【max_steps是什么？停止条件】
max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = 'Cifar_data/cifar-10-batches-bin'

# 这个函数的主要作用是创建一个带有权重衰减的变量。【随机赋予权重的过程？w1是什么？】
# 权重衰减是一种正则化技术，用于减少模型的过拟合风险，通过在损失函数中加入权重的L2范数的平方来实现
# 计算每个权重相应的loss

def variable_with_weight_loss(shape, stddev, w1):
    # 创建一个形状为 shape，标准差为 stddev 的截断正态分布的变量，并将其赋给 var
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        # 计算变量 var 的L2范数的平方，并乘以权重衰减因子w1。
        # 2.【什么是L2范数的平方？】
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        # 将权重衰减项 weights_loss 添加到名为 'losses' 的集合中。
        tf.add_to_collection('losses', weights_loss)
    return var

# 读取数据文件和测试文件
images_train, labels_train = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
image_test, labels_test = Cifar10_data.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 定义模型输入x和标签y_的占位符。x是图像数据，y_是图像标签
# 创建了一个 x 占位符张量，类型为 tf.float32，[batch_size, 24, 24, 3]指定了x的形状
# batch_size: 指定每个批次中将要输入模型的图像数量。 24, 24, 3: 指定每张图像的高度为24像素，宽度为24像素，通道数为3（RGB彩色图像）。
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层
# 调用函数 variable_with_weight_loss，用来创建一个带有权重衰减的变量。
# shape=[5, 5, 3, 64]: 指定了卷积核的形状，这里是一个5x5大小的卷积核，输入通道数为3（对应RGB颜色通道），输出通道数为64（卷积核数量）。
# stddev=5e-2: 指定了正态分布的标准差，用于初始化卷积核的权重。 3.【stddev是tensorflow内置函数吗？】
# w1=0.0: 指定了权重衰减（正则化）的参数，这里设为0表示不使用额外的 L2 正则化项。
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')   #调用卷积。传入参数，输入数据x，卷积kernel1, [1,1,1,1]步长，padding模式
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 卷积层到全连接层前的reshape操作
# 【池化后输出卷积大小：】
reshape = tf.reshape(pool2, [batch_size, -1])  # -1表示将pool2的三维结构拉直成一维结构。pool2：[batch_size,kh,kw,c]如何转成[batch_size,-1]？不理解这个操作，举个例子？
dim = reshape.get_shape()[1].value  # get_shape()[1].value表示获取reshape之后第二个维度的值。#取[batch_size,-1]第二位的值吗？

# 创建第一个全连接层
# 【为什么是384？stddev=0.04, w1=0.004,是经验值吗？】
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 创建第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 创建第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1 = 0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)


# 计算损失，包括权重参数正则化损失和交叉熵损失
# 【这里一团乱麻，不理解...】
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
# tf.train.AdamOptimizer(1e-3) 创建了一个学习率为 1e-3 的 Adam 优化器实例。
# minimize(loss) 方法设置了一个操作 train_op，用于最小化总体损失 loss。
# 这个操作会计算损失函数关于模型参数的梯度，并且根据这些梯度更新模型参数，以减少损失。
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 函数tf.nn.in_top_k()用来计算输出结果中top_k的准确率，函数默认k值是1，即输出分类准确率最高时的数值
# result的数据格式local4*weight3矩阵相乘得到的n*m矩阵，y_是形状为[batch_size]的占位符，result与y_标签比较，如何比较？
top_k_op = tf.nn.in_top_k(result, y_, 1)
#  tf.global_variables_initializer()在 TensorFlow 中的作用是初始化计算图中的所有全局变量.【该函数中计算图是什么？是之前定义的卷积层和全连接层吗？】
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # tf.train.start_queue_runners() 启动了输入数据的队列线程
    tf.train.start_queue_runners()
    # 每隔100step会计算并展示当前的loss、每秒钟训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        # 执行训练操作 train_op 和损失计算 loss，并通过 feed_dict 提供输入数据。
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time  # 计算每个训练步骤的耗时
        if step % 100 == 0:  # 每隔100步输出一次训练结果
            examples_per_sec = batch_size / duration  # 计算每秒处理的样本数
            sec_per_batch = float(duration)  # 计算没批数据处理时间
            print('step %d, loss = %.2f(%.1f examples/sec; %.3f sec/batch' % (step, loss_value, examples_per_sec, sec_per_batch))

# 计算最终正确率
    num_batch = int(math.ceil(num_examples_for_eval/batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环中统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    print('accuracy = %.3f%%' % ((true_count/total_sample_count)*100))