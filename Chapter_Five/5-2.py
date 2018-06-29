# -*- coding:utf-8 -*-
"""
常用层
"""
# 5.2.1 Dense层（全连接层）

# 5.2.2 activation层(对一个层的输出施加激活函数)
# 当使用activation作为第一层时，要指定input_shape

# 5.2.3 Dropout层
# 为输入数据施加Dropout。Dropout在训练过程中每次更新参数，随机断开一定百分比的输入神经元。防止过拟合。
import keras
# rate：0-1的浮点数，用于控制需要断开的神经元的比例。
# noise_shape：整数张量。seed：随机种子
from keras.layers import Convolution2D, Flatten, Reshape

keras.layers.core.Dropout(rate=0.1, noise_shape=None, seed=None)

# 5.2.4 Flatten 层
# 用来将输入压平，将多维的输入一维化。多用于卷积层到全连接层的过渡。不影响batch的大小。
keras.layers.core.Flatten()
# 例子(卷积神经网络CNN)
from keras.models import Sequential
model = Sequential()
model.add(Convolution2D(64, 3, 3,
          border_mode='same',
          input_shape=(3, 32, 32)))
model.add(Flatten())

# 5.2.5 Reshape层
# 将输入的shape转为特定的shape
# 例
model = Sequential()
model.add(Reshape((3, 4), input_shape=(12, )))
model.add(Reshape((6, 2)))
model.add(Reshape((-1, 2, 2)))

# 5.2.6 Permute层
# 将输入的维度按照给定模式进行重排。当需要RNN与CNN连接时，可能会用到该层。

