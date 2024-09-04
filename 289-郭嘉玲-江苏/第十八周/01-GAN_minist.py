from __future__ import print_function, division
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # 构建并编译判别器模型，使用二元交叉熵作为损失函数，Adam优化器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        # 构建生成器模型
        self.generator = self.build_generator()

        # 输入层
        z = Input(shape=(self.latent_dim, ))
        img = self.generator(z)
        # 训练生成器时判别器权重不更新
        self.discriminator.trainable = False
        # 对生成的图像进行评分，判断其真实性
        validity = self.discriminator(img)
        # 创建一个组合模型将生成器和判别器堆叠起来，用于训练生成器欺骗判别器
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 定义生成器的构建函数
    def build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        # 打印生成器模型摘要
        model.summary()
        # 创建并返回生成器模型，接受潜在空间噪声作为输入，生成图像作为输出
        noise = Input(shape=(self.latent_dim, ))
        img = model(noise)
        return Model(noise, img)

    # 定义判别器的构建函数
    def build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        # 创建并返回判别器模型
        img = Input(shape=self.img_shape)
        validity = model(img)
        # 接受图片作为输入，输出真实值概率
        return Model(img, validity)

    # 定义训练函数。设置训练轮数、批大小、样本保存间隔（监控网络的训练进度）
    def train(self, epochs, batch_size=128, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()  # 加载数据，返回训练数据的特征和标签，返回测试数据的特征和标签
        X_train = X_train / 127.5-1.  # 归一化处理
        X_train = np.expand_dims(X_train, axis=3)  # 在数组第三维度增加一个维度
        # 定义真实和虚假的标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 开始循环训练
        for epoch in range(epochs):

            # 从训练集中选择一个随机批次的图像，生成一批新图像
            idx = np.random.randint(0, X_train.shape[0], batch_size)  # 生成随机索引。整数范围：0到X_train.shape[0],数量batch_size个
            imgs = X_train[idx]  # 选择样本
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # 训练判别器，分别计算真实图像和生成图像损失并取平均
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # 生成新的噪声训练生成器
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            print("%d [D loss: %f, acc.: %.2f%%][G loss : %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # 生成一组图像并保存至文件中，用于查看训练进度
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r*c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("./mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=10000, batch_size=32, sample_interval=200)
