# -*- coding:utf-8 -*-
"""
文本预处理
"""
# 6.2.1 分割句子获得单词序列
from keras.preprocessing.text import Tokenizer,one_hot,text_to_word_sequence
import numpy as np

def test_text_to_word_sequence():
    sequence = text_to_word_sequence('Learn Git and GitHub without any code')
    # print(sequence)

# 6.2.2 OneHot 序列编码器
"""
记录词在字典中的下标。
返回值：整数列表，每个整数是[1, n]之间的值，代表一个单词（不保证唯一性，如果字典长度不够，不同的单词可能被编为同一个码。）
"""
def test_noe_hot():
    text = 'Learn Git and GitHub without any code'
    # 这里5是指：字典长度
    encoded = one_hot(text, 5)
    # print(encoded)

# 6.2.3 单词向量化
def test_tokenizer():
    texts = ['The cat sat on the mat',
             'The dog sat on the log',
             'Dogs and cats living together']
    # num_words:处理的最大单词数量。被限制处理数据集中最常见的n个单词
    tokenizer = Tokenizer(num_words=10)
    tokenizer.fit_on_texts(texts)
    print('word_counts: ', tokenizer.word_counts) # 在训练期间出现的次数
    print('word_docs: ', tokenizer.word_docs) # 在训练期间，单词出现在了几份文档中
    print('word_index: ', tokenizer.word_index) # 排名索引
    print('document_count: ', tokenizer.document_count) # 被训练的文档数

    # 测试序列生成器
    sequences = []
    for seg in tokenizer.texts_to_sequences_generator(texts):
        sequences.append(seg)

    # 测试文本序列以矩阵的形式表达特征
    tokenizer.fit_on_sequences(sequences)
    for mode in ['binary', 'count', 'tfidf', 'freq']:
        matrix = tokenizer.texts_to_matrix(texts, mode)
        print(mode, " : ", matrix)