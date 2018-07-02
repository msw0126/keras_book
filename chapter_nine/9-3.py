# -*- coding:utf-8 -*-
"""
实现DCGAN生成对抗网络还原和模仿MNIST样本
参考文档：https://blog.csdn.net/gjq246/article/details/75118751
         https://blog.csdn.net/u013948010/article/details/78580743
"""

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt


class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
        time.sleep(10)

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))


class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None # discriminator
        self.G = None # generator
        self.AM = None # adversarial model
        self.DM = None # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 * 28 * 1, depth = 1
        # Out: 10 * 10 * 1, depth = 64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        # 二维卷积层，即对图像的空域卷积。
        self.D.add(Conv2D(depth * 1, # 卷积核的数目（即输出的维度）
                          5, # 单个整数或由两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，则表示在各个空间维度的相同长度。
                          strides=2, # 单个整数或由两个整数构成的list/tuple，为卷积的步长。如为单个整数，则表示在各个空间维度的相同步长。
                          input_shape=input_shape, # 输入张量的shape
                          padding='same', # 补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
                          activation=LeakyReLU(alpha=0.2))) # 激活函数。每一层的网络输出都要经过激活函数
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 2,
                          5,
                          strides=2,
                          padding='same',
                          activation=LeakyReLU( alpha=0.2 )))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 4,
                          5,
                          strides=2,
                          padding='same',
                          activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth * 8,
                          5,
                          strides=1,
                          padding='same',
                          activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten()) # 压平。常用在卷积层与全连接层的中间
        self.D.add(Dense(1)) # 1：输出维度
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim * dim * depth
        self.G.add(Dense(dim*dim*depth, input_dim=100)) # input_dim:可以指定输入数据的维度
        # 将前一层的激活输出按照数据batch进行归一化。
        self.G.add(BatchNormalization(momentum=0.9)) # momentum是梯度下降法中一种常用的加速技术。
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim * dim *depth
        # Out: 2*dim x 2*dim x depth/2
        # 将数据的行和列分别重复size[0]和size[1]次
        self.G.add(UpSampling2D())
        # 该层是转置的卷积操作（反卷积）。需要反卷积的情况通常发生在用户想要对一个普通卷积的结果做反方向的变换。
        # 例如，将具有该卷积层输出shape的tensor转换为具有该卷积层输入shape的tensor。同时保留与卷积层兼容的连接模式。
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        self.DM = Sequential()
        # 该优化器通常是面对递归神经网络时的一个良好选择
        optimizer = RMSprop(lr=0.0008, # 大或等于0的浮点数，学习率
                            clipvalue=1.0, #
                            decay=6e-8) # 大或等于0的浮点数，每次更新后的学习率衰减值
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy']) # 性能评估。
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0004,
                            clipvalue=1.0,
                            decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer,
                        metrics=['accuracy'])
        return self.AM


class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist",
                                                 one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1,
                                            self.img_rows,
                                            self.img_cols,
                                            1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator = self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval > 0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])  # 外层是16个元素，每个元素有100个元素
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                                                          self.x_train.shape[0],
                                                          size=batch_size ), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-0.1, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval > 0:
                if (i + 1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],
                                     noise=noise_input,
                                     step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]
        plt.figure(figsize=(10, 10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
        else:
            plt.show()


if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train( train_steps=10000, batch_size=256, save_interval=500 )
    timer.elapsed_time()
    mnist_dcgan.plot_images( fake=False, save2file=True )