# 使用keras实现手写数字识别

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)

# 打印第一张图片
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)  # 图像显示为黑白色调
plt.show()

# 使用keras搭建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 数据预处理
from tensorflow.keras.utils import to_categorical
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
# to_categorical函数将整数标签转换为独热编码形式
print('before change:', test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('after change:', test_labels[0])

# 数据输入网络进行训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)  # 输出模型训练效果
print(test_loss, 'test_acc', test_acc)

# 模型测试
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))  # 数据展开为一维的（784，）
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print('the number for the picture is :', i)
        break
