# -*- coding:utf-8 -*-

"""
词向量文本分类
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
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

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

texts = []
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
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
