# -*- coding:utf-8 -*-

import numpy as np

a = np.array([[[1,2], [3,4], [5,6]], [[1,2], [3,4], [5,6]]])
sum0 = np.sum(a, axis=0)
sum1 = np.sum(a, axis=1)
sum2 = np.sum(a, axis=2)
"""
axis=0 是指在这个多维列表里的最里层
axis=1 是指在这个多维列表的从里到外的第二层
"""
print("-----------")
print(sum0)
print("-------------")
print(sum1)
print("-----------------")
print(sum2)