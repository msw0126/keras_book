# -*- coding:utf-8 -*-
"""
训练一个准确率高于90%的Cifar-10预测模型
"""
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation,Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D, GlobalMaxPooling2D
from chapter_nine.lsuv_init import LSUVinit

batch_size = 32
num_classes = 10
epochs = 1600
data_augmentation = True

# 数据预处理
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape: ', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# load 得到的数据是随机打乱的。to_categorical：设置成总类别数长度的数组，便于训练计算。
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# print(x_train.shape[1:])
# 训练
model = Sequential()
# Conv2D:二维卷积层。filters:卷积核的数目。kernel_size:单个整数或两个整数构成的list/tuple，卷积核的宽度和长度。如为单个整数，表示在各个空间维度的相同长度。
# padding：补0策略，为“valid”, “same” 。“valid”代表只进行有效的卷积，即对边界数据不处理。“same”代表保留边界处的卷积结果，通常会导致输出shape与输入shape相同。
# 当使用该层作为第一层时，应提供input_shape参数。
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3),
                 padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(80, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(GlobalMaxPooling2D())
model.add(Dropout(0.25))

model.add(Dense(500))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 选择各方面表现较好的Adam优化器，学习率选择建议的默认值0.0001，学习率衰减为1e-6
opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

"""
模型编译阶段，
我们使用常用的类别交叉熵损失函数，
指定之前定义的Adam优化器
keras内置的准确率性能评估函数，保证训练时得到模型基本的性能评估情况
"""
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 使用LSUVinit类初始化模型待训练参数
model = LSUVinit(model, x_train[:batch_size, :, :, :])
# 指定keras内置的TensorBoard回调函数，保证训练中实时对性能评估进行可视化。
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True)

if not data_augmentation:
    print("Not use data augmentation")
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=[tbCallBack])
else:
    print('Using real-time data augmentation')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                                    batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test), callbacks=[tbCallBack])
