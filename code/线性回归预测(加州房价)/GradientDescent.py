from numpy.lib.shape_base import apply_along_axis, column_stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder


# 画图
def Show(x, y):
    x = np.array(x.T)
    y = np.array(y.T)
    line_x = np.linspace(0, 600000, 2)
    line_y = line_x
    x=x*data_max[8]
    y=y*data_max[8]
    
    plt.rcParams['font.family'] = ['STSong']
    plt.scatter(x, y, 10, c='r', label='预测值', marker='o')
    plt.plot(line_x, line_y, ls='-', lw=2, label='真实值', color='g')
    plt.legend(loc='best')
    plt.xlabel('真实房价', fontsize=15)
    plt.ylabel('预测房价', fontsize=15)
    plt.show()


# R2评估
def R2(y_test, y_true):
    return 1 - ((y_test - y_true) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()


# 读取数据
data = pd.read_csv('housing.csv')
data = data.dropna(axis=0, how='any')

# one-hot
OneHot_data = pd.get_dummies(data[['ocean_proximity']])
data = data.join(OneHot_data)
data = data.drop(['ocean_proximity'], axis=1)
data_max=data.max(axis=0)
data = data / data.max(axis=0)  # 归一化，防止溢出

# 划分训练集和测试集 
train_data = data.sample(frac=0.7, random_state=0, axis=0)
test_data = data[~data.index.isin(train_data.index)]  # 逆函数，剔除训练集
# 取出训练集的X & Y
train_data_Y = train_data[['median_house_value']]  # 取出Y
train_data_X = train_data.drop(['median_house_value'], axis=1)  # 取出X

# 定义学习率α，和迭代次数
alpha = 0.01
learning_count = 100000

# 训练DataFrame→→→np.array
X = train_data_X.values
Y = train_data_Y.values

W = np.zeros(shape=[1, X.shape[1]])  # (1,20640)
b = 0

for i in range(learning_count):
    # y_h 是预测的y值
    y_h = np.dot(X, W.T) + b
    # 求出损失
    lost = y_h - Y
    W = W - alpha * (1 / len(X)) * np.dot(lost.T, X)
    # 给 ω 做梯度下降
    b = b - alpha * (1 / len(X)) * lost.sum()
    # 代价
    cost = (lost ** 2) / (len(X))

# 对训练完的模型进行评估

test_data_Y = test_data[['median_house_value']]  # 取出Y
test_data_X = test_data.drop(['median_house_value'], axis=1)  # 剔除Y

# 测试DataFrame→→→np.array
test_data_Y = test_data_Y.values
test_data_X = test_data_X.values
test_data_Y_pred = np.dot(test_data_X, W.T) + b

print('R²:')
print(R2(test_data_Y_pred, test_data_Y))

Show(test_data_Y, test_data_Y_pred)
