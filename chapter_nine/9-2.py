# -*- coding:utf-8 -*-

"""
词向量文本分类
卷积层与池化层的区别：https://blog.csdn.net/liulina603/article/details/47727277
"""

from __future__ import print_function
import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = r"F:/Temp/"
GLOVE_DIR = BASE_DIR + "/glove.6B/"
TEXT_DATA_DIR = BASE_DIR + "/20_newsgroup/"
MAX_SEQUENCE_LENGTH = 1000 # 对每个新闻文本，最多保留的单词数
MAX_NB_WORDS = 20000 # 所有新闻样本总的字典中的单词数
EMBEDDING_DIM = 100 # 词嵌入入层的输出尺寸
VALIDATION_SPLIT = 0.2 # 测试集占到总数据集的比例

print("Indexing word vectors.")

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, "glove.6B.100d.txt"), 'r', encoding='UTF-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

print('Processing text dataset')

texts = [] # 是一每篇文章作为一个元素的列表
lables_index = {}
lables = []

for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        lable_id = len(lables_index)
        lables_index[name] = lable_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3, ):
                    f = open(fpath, 'r', encoding='UTF-8')
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                # 找到某段文本下一行是空行的单词数，将这一部分文本删除。这部分文本是解释文本
                i = t.find("\n\n")
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                lables.append(lable_id)
print('Found %s texts.' % len(texts))


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # 得到词索引[[1, 2, 3, 4], [1, 2, 3, 5]]
word_index = tokenizer.word_index # 得到词与索引的字典 {'some': 1, 'thing': 2, 'to': 3, 'eat': 4, 'drink': 5}
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH) # 对长度不足MAX_SEQUENCE_LENGTH的文章用0填充
lables = to_categorical(np.asarray(lables)) # 最后将标签处理成 one-hot 向量，比如 6 变成了 [0,0,0,0,0,0,1,0,0,0,0,0,0]，
print('Shape of data tensor:', data.shape)
print('Shape of lable tensor:', lables.shape)

# 拆分数据为训练数据和测试数据(这一步是把原来的数据顺序打乱)
indices = np.arange(data.shape[0]) # 得到array([0， 1 ...19997])
np.random.shuffle(indices) # 打乱顺序
data = data[indices]
lables = lables[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = lables[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = lables[-num_validation_samples:]


print('Preparing embedding matrix.')

num_words = min(MAX_NB_WORDS, len(word_index)) # 单词数
embeddings_matrix = np.zeros((num_words, EMBEDDING_DIM)) # 得到一个20000 * 100de 矩阵,20000是总的单词数，100是词向量的维度
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

# 词向量矩阵加载到embedding层,trainable=False 是这个编码层不可再训练。
embedding_layer = Embedding(num_words, # 大或等于0的整数，字典长度，即输入数据最大下标+1
                            EMBEDDING_DIM, # 大于0的整数，代表全连接嵌入的维度
                            weights=[embeddings_matrix],
                            input_length=MAX_SEQUENCE_LENGTH, # 当输入序列的长度固定时，该值为其长度。
                            trainable=False)

print('Training Model.')

sequences_input = Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')
embedded_sequences = embedding_layer(sequences_input)
x = Conv1D(256, # 卷积核的数目（即输出的维度）
           5, # 整数或由单个整数构成的list/tuple，卷积核的空域或时域窗长度
           activation='relu' # 激活函数，为预定义的激活函数名
           )(embedded_sequences)
# 对时域1D信号进行最大值池化.5是池化窗口大小.(通过池化来降低卷积层输出的特征向量，同时改善结果)
x = MaxPooling1D(5)(x)
x = Dropout(0.4)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Dropout(0.4)(x)
x = Conv1D(256, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Dropout(0.25)(x)
x = Flatten()(x) # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
x = Dense(128, activation='relu')(x)
preds = Dense(len(lables_index), activation='softmax')(x)

model = Model(sequences_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=64,
          epochs=10,
          validation_data=(x_val, y_val))

# 按batch计算在某些输入数据上的模型误差
score = model.evaluate(x_val, y_val, verbose=0)

# 输出训练好的模型在测试集上的表现
print('Test score:', score[0])
print('Test accuracy:', score[1])



