# -*- coding:utf-8 -*-

# 导入顺序模型
from keras.models import Sequential
# 导入全连接层Dense， 激活层Activation 以及 Dropout层
from keras.layers.core import Dense, Dropout, Activation

"""
Sequential模型是多个网络层的线性堆叠，可以向Sequential模型传递一个Layer List构造此模型。
"""
model = Sequential([
    Dense(32, units=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax')
])

"""
也可以通过add()方法，依次将Layer添加到模型中。
"""
model = Sequential()
# Dense（全连接层）units：大于0的整数，代表该层的输出维度。
model.add(Dense(32, units=784))
# Relu：近似生物神经激活函数，最近出现的。
# 参考文档：https://www.jianshu.com/p/74edd3c9f593
model.add(Activation('relu'))

"""
下面的两个指定输入数据shape的方法是严格等价的
"""
model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model = Sequential()
model.add(Dense(32, input_shape=(784)))


# 模型编译
"""
compile接收三个参数：
optimizer：优化器。该参数可指定为已预定义的优化器名，如rmsprop、adagrad或Optimizer类的对象
loss：损失函数。该参数是模型师徒最小化的目标函数，它可设为预定义的损失函数名，如categorical_crossentropy、mse，也可以设为一个损失函数。
metrics：性能评估指标列表。对于分类问题，我们一般设为 metrics=['accuracy']
"""
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
