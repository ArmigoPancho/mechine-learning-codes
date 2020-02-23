# 代码是得到HMM的前向与后向算法的简单实现
import numpy as np

def Forward(trainsition_probability,emission_probability,pi,obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是发射矩阵（观测状态矩阵）
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability  = np.array(emission_probability)
    # print(emission_probability[:,0])
    pi = np.array(pi)
    Row = np.array(trainsition_probability).shape[0]

    F = np.zeros((Row,Col))                      #最后要返回的就是F，就是我们公式中的alpha
    F[:,0] = pi * np.transpose(emission_probability[:,obs_seq[0]])  #这是初始化求第一列,就是初始的概率*各自的发射概率，是行矩阵
    # print(F[:,0])
    for t in range(1,len(obs_seq)):              # 这里相当于填矩阵的元素值，len(obs_seq)是已观测时序维度（列），Row值是隐层状态的维度(行)，比如这里有三个盒子可以抽取。
        for n in range(Row):                     # n是代表某个隐藏状态
            F[n,t] = np.dot(F[:,t-1],trainsition_probability[:,n])*emission_probability[n,obs_seq[t]]   # obs_seq[t]表示t时序摸出某个球，emission_probability[n,obs_seq[t]表示所某个盒子的已观测球的观测状态输出概率
            # print('F['+ str(n) +','+ str(t) +']=',F[n,t])
    print('前向算法：\n',F)
    return F

def Backward(trainsition_probability,emission_probability,pi,obs_seq):
    """
    :param trainsition_probability:trainsition_probability是状态转移矩阵
    :param emission_probability: emission_probability是观测状态概率矩阵
    :param pi: pi是初始状态概率
    :param obs_seq: obs_seq是已观察状态序列
    :return: 返回结果
    """
    trainsition_probability = np.array(trainsition_probability)
    emission_probability = np.array(emission_probability)
    pi = np.array(pi)                 #要进行矩阵运算，先变为array类型

    Row = trainsition_probability.shape[0]
    Col = len(obs_seq)
    F = np.zeros((Row,Col))
    F[:,(Col-1):] = 1                  #最后的每一个元素赋值为1

    for t in reversed(range(Col-1)):
        for n in range(Row):
            print('F[:,t+1]=',F[:,t+1])
            print('trainsition_probability[n,:]=',trainsition_probability[n,:])
            print('emission_probability[:,obs_seq[t+1]]=',emission_probability[:,obs_seq[t+1]])
            F[n,t] = np.sum(F[:,t+1]*trainsition_probability[n,:]*emission_probability[:,obs_seq[t+1]])

    return F

if __name__ == '__main__':
    trainsition_probability = [[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]]
    emission_probability = [[0.5,0.5],[0.4,0.6],[0.7,0.3]] # 这里观测状态的列维度是2，对应于obs_seq的维度
    pi = [0.2,0.4,0.4]

    # 然后下面先得到前向算法,在A,B,pi参数已知前提下，求出特定观察的序列的概率是多少?
    obs_seq = [0,1,0]
    Row = np.array(trainsition_probability).shape[0]
    Col = len(obs_seq)

    F = Forward(trainsition_probability,emission_probability,pi,obs_seq)                  #得到前向算法的结果
    F_backward = Backward(trainsition_probability,emission_probability,pi,obs_seq)        #得到后向算法的结果

    res_forward = 0
    for i in range(Row):                         #将最后一列相加就得到了我们最终的结果
        res_forward+=F[i][Col-1]                          #求和于最后一列
    emission_probability = np.array(emission_probability)
    #下面是得到后向算法的结果
    res_backword = 0
    res_backward = np.sum(pi*F_backward[:,0]*emission_probability[:,obs_seq[0]])      # 一定要乘以发射的那一列
    print("res_backward = {}".format(res_backward))
    print("res_forward = {}".format(res_forward))