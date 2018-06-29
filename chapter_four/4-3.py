# -*- coding:utf-8 -*-
"""
函数式模型

Keras 函数式模型接口是用户定义多输出模型、非循环有向模型或具有共享层模型等复杂模型的途径。
当模型需要多一个的输出，应该选择函数式模型。函数是模型是最广泛的一种模型，Sequential模型只是函数式模型的一种特殊情况。
"""

# 4.3.1 全连接网络
"""
层对象接受张量为参数，返回一个张量。
输入是张量，输出也是张量的一个网络就是一个模型，可通过Model定义
这样的模型可以被像keras的Sequential一样来训练。
利用函数式模型的接口，可以很容易的重用已经训练好的模型
"""
from keras.layers import Input, Dense
from keras.models import Model

# This return a tensor
inputs = Input(shape=(784, ))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, lables)

"""
这种方式可以快速的创建能处理序列信号的模型，将一个图像分类的模型变为一个对视频分类的模型，只需要一行代码
"""
x = Input(shape=(784, ))
y = model(x)

from keras.layers import TimeDistributed

input_sequences = Input(shape=(20, 784))
processed_sequences = TimeDistributed(model)(input_sequences)

# 4.3.2 函数模型接口
"""
keras 的函数式模型为Model，即广义的拥有输入和输出的模型。使用Model来初始化一个函数式模型
"""

from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32, ))
b = Dense(32)(a)
model = Model(inputs=a, outputs=b)
# 在这里，模型以a为输入，以b为输出，同样可以构造拥有多输入和多输出的模型
model = Model(inputs=[a1, a2, a3], outputs=[b1, b2, b3])

# 4.3.3 多输入和多输出模型
"""
主要的输入时新闻本身，即一个整数的序列（每个整数编码一个单词）。
这些整数位于1-10000之间（即我们的字典有100000个词）。这个序列有100个单词。
"""
import keras
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model

main_input = Input(shape=(100, ), dtype='int32', name='main_input')
# 嵌入层将正整数（下标）转换为具有固定大小的向量。
# input_dim：大或等于0的整数，字典长度，即输入数据最大下标+1
# output_dim：大于0的整数，代表全连接嵌入的维度
# input_length：当输入序列的长度固定时，该值为其长度。如果要在该层后接Flatten层，然后接Dense层，\
# 则必须指定该参数，否则Dense层的输出维度无法自动推断。
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

# 插入一个额外的损失函数，使得即使主损失很高的情况下，LSTM和Embedding也可以平滑的训练
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# 将LSTM与额外的输入数据串联起来组成输入，送入模型中。
auxiliary_input = Input(shape=(5, ), name='aux_input')
x = keras.layers.concatenate([lstm_out, auxiliary_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# 最后定义两个输入两个输出的模型
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

# 编译模型
# 对额外的损失赋0.2的权重。这里给loss传递单个损失函数。
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              loss_weights=[1., 0.2])

# 训练
model.fit([head_data, additional_data], [lables, lables],
          epochs=50,
          batch_size=32)

# 4.3.4 共享层模型
"""
一条微博最多是140个字符，扩展的ASCII码表编码了常见的256个字符
"""
import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model

tweet_a = Input(shape=(140, 256))
tweet_b = Input(shape=(140, 256))

# 若要对不同的输入共享同一层，就初始化该层一次，然后多次调用它
shared_lstm = LSTM(64)

encoded_a = shared_lstm(tweet_a)
encoded_b = shared_lstm(tweet_b)

# 合并两个向量
merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=1)

# 再加上逻辑回归
predictions = Dense(1, activation='sigmoid')(merged_vector)

model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([data_a, data_b], lables, epochs=10)