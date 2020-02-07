# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 19:01:17 2020

@author: SenBai
"""
import helper as hlp
import numpy as np

#设置随机种子，保证每次结果一致
seed = 42
#获取随机数开始位置
rnd = np.random.RandomState(seed)
print("rnd=",rnd)
#设置数据集大小
N = 40
#设置数据集维度
d = 2
#获取用于制图的参数
a, b, c, X, y, s, w = hlp.f(N, d, rnd)
print("a=",a)
print("b=",b)
print("c=",c)
print("X=",X)
print("y=",y)
print("s=",s)
print("w=",w)
hlp.plot_helper(a, b, c, X, y, s, w)
