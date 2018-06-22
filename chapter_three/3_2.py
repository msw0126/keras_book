# -*- coding:utf-8 -*-

"""
参考文档：https://blog.csdn.net/u012969412/article/details/70882296
          http://keras-cn.readthedocs.io/en/latest/layers/core_layer/
"""
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

# Sequential()是keras中主要的模型。是一系列网络层按顺序构成的栈。
model = Sequential()
# Dense全连接层（对上一层的神经元进行全部连接，实现特征的非线性组合）
# units：大于0的整数，代表该层的输出维度。
# input_shape参考链接文档：https://blog.csdn.net/thinking_boy1992/article/details/53207177
model.add(Dense(512, input_shape=(784, )))
# Activation激活层对一个层的输出施加激活函数
# relu其中一种激活函数。参考文档：https://blog.csdn.net/niuwei22007/article/details/49208643
model.add(Activation('relu'))
# 为输入数据施加Dropout。
# Dropout将在训练过程中每次更新参数时按一定概率（rate）随机断开输入神经元，Dropout层用于防止过拟合。
model.add(Dropout(0.2))

# 编译模型
# loss损失函数配置,categorical_crossentropy（交叉熵-多分类损失函数）相关文档：https://blog.csdn.net/u010412858/article/details/76842216
# optimizer（优化函数）,sgd(随机梯度下降函数)。参考文档：https://blog.csdn.net/zjm750617105/article/details/51321915
model.compile(loss='categorical_crossentropy', optimizer='sgd')

# # 准备数据(此处是错误的数据)
# ratings = pd.read_csv("./data/ratings.dat", sep="::", engine="python", names=['user_id','movie_id','rating','timestamp'])
# users = ratings['user_id'].values
# movies = ratings['movie_id'].values
# X_train = [users, movies]
# y_train = ratings['rating'].values

# 训练模型
model.fit(X_train, y_train, batch_size=100,
          nb_epoch=10, shuffle=True, verbose=1, validation_split=0.2)

#评估模型
model.evaluate(X_train, y_train, batch_size=100, verbose=1)