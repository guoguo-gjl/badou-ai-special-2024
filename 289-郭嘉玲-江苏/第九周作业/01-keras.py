'''
手写数字识别
'''

# 01.导入数据集
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape=', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)

# 02.打印一张测试图片看看
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.imshow()

# 03.神经网络搭建
from tensorflow.keras import models
from tensorflow.keras import layers
# 定义一个变量存储模型结构
# models.Sequential()——串联的模型，之后加入的层都是串联在一起的，上一个层的输出是下一个层的输入
network = models.Sequential()
# .add：在后面加一层
# layers.Dense()构造一个数据处理层
# 隐藏层512个节点，激活函数relu，输入shape是28*28
network.add(layers.Dense(512, activation='relu', input_shape=(28*28)))
# 输出层10个，激活函数softmax，输入是上一层的输出512个
network.add(layers.Dense(10, activation='softmax'))

# 编译阶段。将模型结构存起来
# optimizer 优化框架
# loss 损失函数
# metrics 积分方法
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 04-模型处理
# 数据预处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255  # 像素值归一化
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
# one hot处理
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 05-模型训练
# epochs：代数
# batch——size：训练一批的数量
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 06-统计训练结果
# 用测试数据评价训练结果好坏
# verbose=1 参数控制是否显示评估过程的详细信息
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 07-模型推理
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print('the number for the picture is :', i)
        break