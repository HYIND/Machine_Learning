import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

# 取3行作为3个聚类的初始值
Clustering_means_list=pd.DataFrame([data_X.iloc[20],data_X.iloc[70],data_X.iloc[120]]).copy()
Clustering_means_list=Clustering_means_list.reset_index(drop=True)

# 创建一列初始化的聚类结果Y
Clustering_results=np.zeros(shape=[1,data_X.shape[0]]).astype(int)

# 并入X
data_result=data_X.join(pd.DataFrame(Clustering_results.T,columns=["Class"]),how='right')

count=0
# 循环聚类，直到结果收敛
while True:
    count=count+1
    # result矩阵存储单个样本到三个聚类中心的距离
    result=np.zeros(shape=[1,3]).astype(np.float64)
    #临时存储聚类结果，用于比较该次聚类结果与上一次相比是否发生变化
    Clustering_results_temp=np.zeros(shape=[1,data_X.shape[0]]).astype(int)
    # 循环计算聚类结果
    for i in range(data_X.shape[0]):
        result[0,0]=pow(data_X.iloc[i]-Clustering_means_list.iloc[0],2).sum()
        result[0,1]=pow(data_X.iloc[i]-Clustering_means_list.iloc[1],2).sum()
        result[0,2]=pow(data_X.iloc[i]-Clustering_means_list.iloc[2],2).sum()
        Clustering_results_temp[0,i]=np.argmin(result)
    # 如果该次聚类前后结果没有变化，则判断收敛，停止循环
    if (Clustering_results==Clustering_results_temp).all():
        break
    Clustering_results=Clustering_results_temp
    # 聚类结果发生变化，生成新的data_result
    data_result=data_X.join(pd.DataFrame(Clustering_results_temp.T,columns=["Class"]),how='right')
    # 更新均值
    data_result0=data_result[data_result.Class==0]
    data_result1=data_result[data_result.Class==1]
    data_result2=data_result[data_result.Class==2]
    for i in range(4) :
        Clustering_means_list.iloc[0,i]=data_result0.iloc[:,i].mean()
        Clustering_means_list.iloc[1,i]=data_result1.iloc[:,i].mean()
        Clustering_means_list.iloc[2,i]=data_result2.iloc[:,i].mean()


print('迭代次数：',count)
# 画聚类结果图
plt.subplot(1, 2, 2)
show(data_result)

data_Y=data.Class
# Accuracy指标
Accuracy=metrics.accuracy_score(np.array(data_Y), (Clustering_results)[0])
print("Accuracy:",Accuracy)

# NMI指标
result_NMI=metrics.normalized_mutual_info_score(np.array(data_Y), (Clustering_results)[0])
print("NMI:",result_NMI)

# ARI指标
ARI = metrics.adjusted_rand_score(np.array(data_Y), (Clustering_results)[0])
print("ARI:",ARI)

# 显示结果
plt.show()
