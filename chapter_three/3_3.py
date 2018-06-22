# -*- coding:utf-8 -*-

# 导入mnist数据库， mnist是常用的手写数字库
# 导入顺序模型
from keras.models import Sequential
# 导入全连接层Dense， 激活层Activation 以及 Dropout层
from keras.layers.core import Dense, Dropout, Activation

# 加载数据
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
from keras.utils import np_utils

from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_train, Y_train = mnist_data.train.images, mnist_data.train.labels
X_test, Y_test = mnist_data.test.images, mnist_data.test.labels
X_train = X_train.reshape(-1, 28, 28,1).astype('float32')
X_test = X_test.reshape(-1,28, 28,1).astype('float32')

#打印训练数据和测试数据的维度
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

#修改维度
X_train = X_train.reshape(55000,784)
X_test = X_test.reshape(10000,784)
print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

# keras中的mnist数据集已经被划分成了55,000个训练集，10,000个测试集的形式，按以下格式调用即可

# X_train原本是一个60000*28*28的三维向量，将其转换为60000*784的二维向量

# X_test原本是一个10000*28*28的三维向量，将其转换为10000*784的二维向量

# 将X_train, X_test的数据格式转为float32存储
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
# 打印出训练集和测试集的信息
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 建立顺序型模型
model = Sequential()
"""
模型需要知道输入数据的shape，
因此，Sequential的第一层需要接受一个关于输入数据shape的参数，
后面的各个层则可以自动推导出中间数据的shape，
因此不需要为每个层都指定这个参数
"""

# 输入层有784个神经元
# 第一个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 第二个隐层有512个神经元，激活函数为ReLu，Dropout比例为0.2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 输出层有10个神经元，激活函数为SoftMax，得到分类结果
model.add(Dense(10))
model.add(Activation('softmax'))

# 输出模型的整体信息（打印出模型的概况）
# 总共参数数量为784*512+512 + 512*512+512 + 512*10+10 = 669706
model.summary()

# metrics(指标列表，对应分类问题，一般将该列设置为metrics=['accuracy'])
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
"""
训练模型时指定的参数如下：
batch_size:指定梯度下降时，每个batch包含的样本数。
nb_epoch:训练的轮数，即迭代次数.(现在已改名称为epochs)
verbose:日志显示，0为不在标准输出流输出日志信息，1为数据进度条记录，2为epoch输出一行记录
validation_data：指定验证集

fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证
集的话，也包含了验证集的这些指标变化的情况。
"""
history = model.fit(X_train, Y_train,
                    batch_size = 200,
                    epochs = 20,
                    verbose = 1,
                    validation_data = (X_test, Y_test))

# 按batch计算在某些输入数据上的模型误差
score = model.evaluate(X_test, Y_test, verbose=0)

# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])