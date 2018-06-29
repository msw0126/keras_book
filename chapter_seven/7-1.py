# -*- coding:utf-8 -*-
"""
模型性能评估模块
"""
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential

# 7.1.1 Keras 内置性能评估方法
# 一般类评估、稀疏类评估、模糊评估

# 7.1.2 使用Keras内置性能评估
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 通过字符串来使用域定义的性能评估函数
model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=['mae', 'acc'])

# 自定义一个Theano/TensorFlow 函数并使用
from keras import metrics

model.compile(loss='mean_squared_error',
              optimizer='sgd',
              metrics=[metrics.mae,
                       metrics.categorical_accuracy])

# 自定义性能评估函数
import keras.backend as K

def mean_pred(y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred()])