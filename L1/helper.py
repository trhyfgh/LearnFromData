# -*- coding: utf-8 -*-
"""
Created on Fri Feb 7 19:04:17 2020

@author: SenBai
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

def create_data(N,d, rnd, t):
    #初始化数组X
    X = []
    #初始化w=[2,1]
    w = np.array([2,1])

    while (len(X) < N):
        #x = (1,1] 之间取值的二维数组
         x = rnd.uniform(-1,1,size=(d))
         #如果x与w的点积 超过0.1, 0.1是阈值，这部判断可以取消
         X.append(x)

    #X为X的数组
    X = np.array(X)
    #初始化Y， X为二维数组，与以为数组W点积，如果结果>0，标记为+1， 否则标记为-1
    y = 2 * (X.dot(w) > 0) - 1 

    #为每个x添加一个偏置项，1
    w0 = np.ones((N, 1))
    X = np.c_[w0,X]

    return X, y

def judge(X, y, w):
    """
    判别函数，判断所有数据是否分类完成
    """
    n = X.shape[0]
    num = np.sum(X.dot(w) * y > 0)

    return num == n

def f(N,d,rnd,t=0.1, r=1):
    """
    生成N个d维点（不包括偏置项1），x1+...+xd>=t的点标记为+1，x1+...+xd<=-t的点标记为-1，
    rnd为随机数生成器，形式为rnd = np.random.RandomState(seed)，seed为随机种子
	利用PLA更新，如果r=1，那么按照顺序取点，否则随机取点
    """
    #利用f生成数据集
    X, y = create_data(N, d, rnd,t)
    #记录次数
    s = 0
    #初始化w=[0, 0, 0]
    w = np.zeros(d+1)
    #数据数量
    n = X.shape[0]

    #判断是否所有x点w都与y符号相同，如果不同，就进行修正
    while (judge(X, y, w) == 0):
        #对每个点进行修正
        for i in range(n):
            #打一个修正的点
            xg = X[i,:]
            # print(xg)
            #如果该点x点w 与 y 符号不同
            if(xg.dot(w) * y[i] <= 0) :
                 #修正w(t+1) = wt+ yi*xi
                 w = w + y[i] * xg
                 s = s+1

    #直线方程为w0+w1*x1+w2*x2=0,根据此生成点
    
    a = np.arange(-1, 1, 0.1)
    b = (a * w[1] + w[0]) / (- w[2])
    
    #原直线方程为x1+2x2 = 0
    c = - 2 * a
    
    #返回数据
    return a, b, c, X, y, s, w

def plot_helper(a, b, c, X, y, s, w, t=0):
    """
    作图函数
    """
    #画出图像
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='r', s=1)
    plt.scatter(X[y == -1][:, 1], X[y == -1][:, 2], c='b', s=1)
    plt.plot(a, b, label="("+str(w[0])+")+("+str(w[1])+")x1+("+str(w[2])+")x2=0")
    plt.plot(a, c, label="x1+x2="+str(t))
    plt.title(u"经过"+str(s)+u"次迭代收敛")
    plt.legend()
    plt.show()


    
    
    
   
   
