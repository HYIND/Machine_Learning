from numpy.ma.core import count
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from sklearn import metrics

def show(data_in):
    X=np.array((data_in.SepalLength).T)
    Y=np.array((data_in.PetalLength).T)
    class_list=np.array((data_in.Class).T)
    plt.rcParams['font.family'] = ['STSong']
    plt.scatter(X, Y, 6, c=class_list, marker='o')
    plt.xlabel('SepalLength', fontsize=15)
    plt.ylabel('PetalLength', fontsize=15)


# 读取源数据
data_source=pd.read_csv('iris.data',names=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth","Class"])

# 拷贝
data=data_source.copy()
# 把分类数值化处理
data.Class[data.Class=='Iris-setosa']=0
data.Class[data.Class=='Iris-versicolor']=1
data.Class[data.Class=='Iris-virginica']=2
data.Class=data.Class.astype(int)

# 画分类图
plt.subplot(1, 2, 1)
show(data)

# 取X
data_X=data_source.drop(columns='Class')
X=np.array(data_X)

# 预设权值
w1=0.3
w2=0.4
w3=0.3
# 设初始均值，这里取k-means的聚类结果
mu1=[5.006,3.418,1.464,0.244]
mu2=[5.9016129032,2.7483870968,4.3935483871,1.4338709677]
mu3=[6.85,3.0736842105,5.7421052632,2.0710526316]
# 协方差计算
Cov1=np.cov(X,rowvar=False)
Cov2=Cov1.copy()
Cov3=Cov1.copy()
# 存储响应
R_list=np.zeros(shape=[3,150]).astype(np.float64)
# 计算初始响应
R_list[0]=w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))
R_list[1]=w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))
R_list[2]=w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))
# 计算初始聚类
Clustering_results=np.zeros(shape=[150,1]).astype(int)
for i in range(150):
    Clustering_results[i,0]=np.argmax(R_list[:,i])
# 计算似然函数
L_theta=0
for i in range(150):
    L_theta=L_theta+np.log(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X[i])+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X[i])+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X[i]))

# 不断更新，直到结果收敛
count=0
while True:
    count=count+1
    # 更新权值
    w1=np.average(R_list[0])
    w2=np.average(R_list[1])
    w3=np.average(R_list[2])

    # 更新期望
    mu1=np.sum((R_list[0]*(X.T)).T,axis=0)/np.sum(R_list[0])
    mu2=np.sum((R_list[1]*(X.T)).T,axis=0)/np.sum(R_list[1])
    mu3=np.sum((R_list[2]*(X.T)).T,axis=0)/np.sum(R_list[2])

    # 更新协方差
    Cov1_temp=np.zeros(shape=[4,4])
    Cov2_temp=np.zeros(shape=[4,4])
    Cov3_temp=np.zeros(shape=[4,4])
    for i in range(150):
        Cov1_temp=Cov1_temp+R_list[0,i]*np.dot((X[i]-mu1.reshape(1,4)).T,X[i]-mu1.reshape(1,4))
        Cov2_temp=Cov2_temp+R_list[1,i]*np.dot((X[i]-mu2.reshape(1,4)).T,X[i]-mu2.reshape(1,4))
        Cov3_temp=Cov3_temp+R_list[2,i]*np.dot((X[i]-mu3.reshape(1,4)).T,X[i]-mu3.reshape(1,4))
    Cov1=Cov1_temp/np.sum(R_list[0])
    Cov2=Cov1_temp/np.sum(R_list[1])
    Cov3=Cov1_temp/np.sum(R_list[2])

    # 计算响应
    R_list[0]=w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))
    R_list[1]=w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))
    R_list[2]=w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X)/(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X)+w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X)+w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X))

    # 聚类
    for i in range(150):
        Clustering_results[i,0]=np.argmax(R_list[:,i])

    # 计算似然函数
    L_theta_temp=0
    for i in range(150):
        L_theta_temp=L_theta_temp+np.log(w1*scipy.stats.multivariate_normal(mu1, Cov1).pdf(X[i])+
        w2*scipy.stats.multivariate_normal(mu2, Cov2).pdf(X[i])+
        w3*scipy.stats.multivariate_normal(mu3, Cov3).pdf(X[i]))

    # 判断是否收敛
    if abs(L_theta-L_theta_temp)<0.01 :
        break
    L_theta=L_theta_temp

print("迭代次数:",count)

# 合成结果
data_result=data_X.join(pd.DataFrame(Clustering_results,columns=["Class"]),how='right')

# 画聚类结果图
plt.subplot(1, 2, 2)
show(data_result)

data_Y=data.Class

# Accuracy指标
Accuracy=metrics.accuracy_score(np.array(data_Y), (Clustering_results.T)[0])
print("Accuracy:",Accuracy)

# NMI指标
result_NMI=metrics.normalized_mutual_info_score(np.array(data_Y), (Clustering_results.T)[0])
print("NMI:",result_NMI)

# ARI指标
ARI = metrics.adjusted_rand_score(np.array(data_Y), (Clustering_results.T)[0])
print("ARI:",ARI)

# 显示结果
plt.show()