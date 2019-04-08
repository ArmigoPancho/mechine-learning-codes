from numpy import *
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

#  Linear Discriminant Analysis 线性判别分析
#    LDA的编程主要参考《机器学习》书上P62的3.39 以及P61的3.33这两个式子。
df = pd.read_csv('watermelon_3a.csv')  #文件要另外保存为csv格式后缀。

def calulate_w():
    df1 = df[df.label == 1]  #把label1=1行全部选出来赋值给df1
    df2 = df[df.label == 0]  ##把label1=1行全部选出来赋值给df2
    X1 = df1.values[:, 1:3]  #样本为1的density,ratio_sugar两列数值，无列名
    X0 = df2.values[:, 1:3]  #类似上面
    mean1 = array([mean(X1[:, 0]), mean(X1[:, 1])])   #均值向量1[ 0.57375  0.27875] ，前一个是（类别为1）密度的均值，后一个是含糖率的均值。
    mean0 = array([mean(X0[:, 0]), mean(X0[:, 1])])   #类别为0的均指向量
    m1,m2 = shape(X1)                         # m1 = 8个1类的样本数，这shape()就是取行号。shape(X1)=(8, 2)
    sw = zeros(shape=(2, 2))                  # 初始化类内散度矩阵
    for i in range(m1):
        xsmean = mat(X1[i, :] - mean1)        #x - μ,样本1的x和均值差值组成的8x2矩阵
        sw += xsmean * xsmean.transpose()     #transpose（）矩阵元素的行列变换，内部和外部维度变化，不传参数，就是转制。

    m0,m1 = shape(X0)
    for i in range(m0):
        xsmean = mat(X0[i, :] - mean0)
        sw += xsmean.transpose() * xsmean   #Sw继续加入第二个类内散度
    w = (mean0 - mean1) * (mat(sw).I)      #(mat(sw).I)就是逆举证
    return w

w = calulate_w()
print('w=',w)   #  w= [[-0.146509882 -0.73871557]]

def plot(w):
    dataMat = array(df[['density', 'ratio_sugar']].values[:, :])
    labelMat = mat(df['label'].values[:]).transpose()  #把类别转制竖过来
    m,m2 = shape(dataMat)                          #  m = 17
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if labelMat[i] == 1:
            xcord1.append(dataMat[i, 0])         #存储好瓜的密度，为一个list
            ycord1.append(dataMat[i, 1])         #存储好瓜的含糖率，也为一个list
        else:
            xcord2.append(dataMat[i, 0])        #下面是坏瓜的density,ratio_sugur
            ycord2.append(dataMat[i, 1])
    plt.figure(1)                               #就是图标的窗口名字
    ax = plt.subplot(111)                       #将画布分割成1行1列，图像画在从左到右从上到下的第1块,这里就是原来的一整块
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')   #坐标，大小，颜色，marker是散点形状,s是正方形，代表好瓜散点。 参考 http://www.cnblogs.com/shanlizi/p/6850318.html
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-0.2, 0.8, 0.1)                  # 横坐标x范围：[-0.2,0.8),增长步伐为0.1
    y = array((-w[0, 0] * x) / w[0, 1])         # w的第一个list参数的第一个-0.14650982是权值w,[[-0.14650982 -0.73871557]]
    print(shape(x))

    print(shape(y))
    plt.sca(ax)                                 #选择图表ax的子图ax
    # plt.plot(x,y)      #ramdomgradAscent
    plt.plot(x, y)  # gradAscent                #梯度上升
    plt.show()
    #plt.xlabel('density')                       #x坐标名称
    #plt.ylabel('ratio_sugar')                   #y坐标名称
    #plt.title('LDA')                            #图表标题
    plt.show()                                  #展示出图


w = calulate_w()
plot(w)

###############

"""
# 完整数据
X = df[:, 1:3]
y = df[:, 3]
# 去掉14行的数据
X = np.delete(X, 14, 0)
y = np.delete(y, 14, 0)
# 获取均值向量
u = []
for i in range(2):
    u.append(np.mean(X[y == i], axis=0))
# 计算类内散度矩阵
m, n = np.shape(X)
Sw = np.zeros((n, n))
for i in range(m):
    x_temp = X[i].reshape(n, 1)
    if y[i] == 0:
        u_temp = u[0].reshape(n, 1)
    if y[i] == 1:
        u_temp = u[1].reshape(n, 1)
    Sw += np.dot((x_temp - u_temp), (x_temp - u_temp).T)
# 利用奇异值分解求Sw的逆
Sw = np.mat(Sw)
U, sigma, V = np.linalg.svd(Sw)
Sw_inv = V.T * np.linalg.inv(np.diag(sigma)) * U.T
# 计算w
w = np.dot(Sw_inv, (u[0] - u[1]).reshape(n, 1))
print(w)
# 画出LDA
f1 = plt.figure(1)

plt.xlim(-0.2, 1)
plt.ylim(-0.5, 0.7)

p0_x0 = -X[:, 0].max()
p0_x1 = (w[1, 0] / w[0, 0]) * p0_x0
p1_x0 = X[:, 0].max()
p1_x1 = (w[1, 0] / w[0, 0]) * p1_x0

print((w[1, 0] / w[0, 0]))

plt.title("watermelon")
plt.xlabel("density")
plt.ylabel("sugar_ratio")
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='bad')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='good')
plt.legend(loc='upper right')

plt.plot([p0_x0, p1_x0], [p0_x1, p1_x1])


# 求投影点
def GetProjectivePoint_2D(point, line):
    a = point[0]
    b = point[1]
    k = line[0]
    t = line[1]

    if k == 0:
        return [a, t]
    elif k == np.inf:
        return [0, b]
    x = (a + k * b - k * t) / (k * k + 1)
    y = k * x + t
    return [x, y]


m, n = np.shape(X)
for i in range(m):
    x_p = GetProjectivePoint_2D([X[i, 0], X[i, 1]], [w[1, 0] / w[0, 0], 0])
    if y[i] == 0:
        plt.plot(x_p[0], x_p[1], 'ko', markersize=5)
    if y[i] == 1:
        plt.plot(x_p[0], x_p[1], 'go', markersize=5)
    plt.plot([x_p[0], X[i, 0]], [x_p[1], X[i, 1]], 'c--', linewidth=0.3)
    plt.show()

"""