from numpy.lib.function_base import average
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import math
import matplotlib.pyplot as plt

# 将中间值sigmoid化
def sigmoid(X):
    data_out = X.copy()
    for i in range(X.shape[0]):
        inx = X[i, 0]
        if inx >= 0:  #对sigmoid函数的优化，避免了出现极大的数据溢出
            sigmoid_data = 1.0 / (1 + np.exp(-inx))
        else:
            sigmoid_data = np.exp(inx) / (1 + np.exp(inx))
        data_out[i,0]=sigmoid_data
    return data_out

#  对sigmoid化后的值进行分类处理
def classify(pred_sigmoid,threshold):
    data_out=pred_sigmoid.copy()
    for i in range(pred_sigmoid.shape[0]):
        inx=pred_sigmoid[i , 0]
        if inx>=threshold:
            data_out[i,0]=1
        else :
            data_out[i,0]=0
    return data_out.astype(np.int64)

# 绘制roc曲线图
def show_roc(fpr,tpr,roc_auc):
    lw = 2
    plt.figure(1,figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) # 假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

# 绘制预测图
def show(pred, true):
    pred = np.array(pred.T)
    true = np.array(true.T)
    count = np.arange(true.shape[1])
    plt.figure(2)
    plt.rcParams['font.family'] = ['STSong']
    plt.ylim(-0.5, 1.5)
    plt.yticks([0, 1], ['<=50K', '>50K'])
    plt.scatter(count, true, 6, c='r', label='真实值', marker='o')
    plt.scatter(count, pred, 6, c='g', label='预测值', marker='o')
    plt.legend(loc='best')
    plt.xlabel('测试数据编号', fontsize=15)
    plt.ylabel('类别', fontsize=15)

# 准确率统计
def deviation(pred, true):
    pred = np.array(pred.T)
    true = np.array(true.T)
    count = 0
    for i in range(true.shape[1]):
        if true[0, i] == pred[0, i]:
            count = count + 1
    return count / true.shape[1]

# 读取数据
data_source = pd.read_csv('adult.csv')

# 归一化处理
int_columns = data_source.dtypes=="int64"
int_columns = list(int_columns[int_columns].index)
for col_name in int_columns:
    Scaler = MinMaxScaler(feature_range=(-1, 1))
    col_value = np.array(data_source[col_name]).reshape(-1,1)
    new_col = Scaler.fit_transform(col_value)
    data_source[col_name] = new_col

# 取X、Y
data_source_X = data_source.drop(columns='income')
data_source_Y = data_source[['income']]

# 取X部分进行One Hot处理
data_X = pd.get_dummies(data_source_X)

# 取Y部分，把类别进行数值化处理
data_Y = data_source_Y
data_Y.income[data_Y.income == '>50K'] = 1
data_Y.income[data_Y.income == '<=50K'] = 0
data_Y['income'] = pd.to_numeric(data_Y['income'])

# 合成处理结果
data = data_X.join(data_Y)

# 取训练集和测试集
train_data = data.sample(frac=0.7, replace=False, axis=0, random_state=1)
test_data = data[~data.index.isin(train_data.index)]

# 取训练集的X、Y
train_data_X = train_data.drop(columns='income')
train_data_Y = train_data[['income']]

# 取测试集的X、Y
test_data_X = test_data.drop(columns='income')
test_data_Y = test_data[['income']]

#设置梯度下降参数
theta = np.zeros(shape=[1,train_data_X.shape[1]])
b = 0
alpha = 0.0005
learn_count = 10000

X=train_data_X.values.astype(np.float64)
Y=train_data_Y.values.astype(np.float64)

# 定义代价函数
def costfunction(theta, x, y):
    y_pred = np.dot(x, theta.T)
    # p = np.exp((y_pred +b) /((np.exp(y_pred) + b)+1))
    p = 1. /(np.exp(-(y_pred + b))+1)
    return -(np.dot(y.T,np.log(p + 1e-6))+ np.dot((1 - y).T , np.log(1 - p + 1e-6)))/len(y)

# 迭代
for i in range(learn_count):
    # 求theta、b的导数
    t=np.dot(X,theta.T) + b
    p = 1. /(np.exp(-t) + 1)
    #p = np.exp(t) /(np.exp(t)+1)
    gd_theta = np.dot(X.T, p - Y)
    gd_b = p - Y
    
    # 对theta和b做梯度下降
    theta = theta - (alpha * gd_theta.T)
    b = b - (alpha * average(gd_b.T))
    
    # 求损失
    cost = costfunction(theta, X, Y)
    #if (i+1)%1000==0:
        #print(cost)

print(theta)
print(b)
# 使用测试样本预测估计结果的中间值
test_data_pred_intermediate = np.dot(test_data_X, theta.T) + b

# 将中间值sigmoid化
test_data_pred_sigmoid = sigmoid(test_data_pred_intermediate)

# 将sigmoid化后的结果按阈值分类
mythreshold=0.5
test_data_pred_classify=classify(test_data_pred_sigmoid,mythreshold)

# 统计准确率
print(deviation(test_data_pred_classify, test_data_Y))

#绘制预测图
show(test_data_pred_classify, test_data_Y)

# 计算真正率和假正率
fpr,tpr,threshold = roc_curve(test_data_Y, test_data_pred_sigmoid)
#计算auc的值
roc_auc = auc(fpr,tpr)

#绘制roc曲线图
show_roc(fpr,tpr,roc_auc)

#显示图
plt.show()