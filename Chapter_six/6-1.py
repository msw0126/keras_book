# -*- coding:utf-8 -*-

# 6.1.1 序列数据填充
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.sequence import skipgrams


def test_pad_sequences():
    a = [[1], [1, 2], [1, 2, 3]]
    # 测试默认填充
    # maxlen:序列的最大长度。大于此长度的序列将被截短。
    # padding:取值为pre或post。表示是在序列的起始还是结尾补。
    b = pad_sequences(a,
                      maxlen=3,
                      padding='pre',)
    print(b)
    print("--------------")
    # 测试裁剪
    b = pad_sequences(a, maxlen=2, truncating='pre')
    print(b)
    print("---------------")
    b = pad_sequences(a, maxlen=2, truncating='post')
    print(b)

    print("------------------")
    # 测试填充指定的特定值(默认值是0)
    b =pad_sequences(a, maxlen=3, value=1)
    print(b)

# 执行函数
# test_pad_sequences()

# 6.1.2 提取序列跳字样本
"""
默认窗口为4.采样所有的窗口内的邻近单词对。（第一个单词"The"不考虑，所以"0"不出现在任何相关单词对中。）
"""
def my_test_skipgrams():
    couples, lables = skipgrams([0, 1, 2, 3], vocabulary_size=4)
    print("couples:", couples)
    print("lables:", lables)

# my_test_skipgrams()

# 6.1.3 生成序列抽样概率表

