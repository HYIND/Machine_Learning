import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 闭式求解
def close(X, y):
    X = X.values
    y = y.values

    # feature左加全1列
    # one_y = np.ones((X.shape[0], 1))  # 补全1的一列
    # X = np.concatenate((one_y, X), axis=1)  # 连接 20433*1 + 20433*14

    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


# 预测值
def pred(X, theta):  # X:(test, 14), theta:(14, 1)
    X = X.values
    # one_y = np.ones((X.shape[0], 1))  # 补全1的一列
    # X = np.concatenate((one_y, X), axis=1)  # 连接 13*1 + 13*14
    Y = np.dot(X, theta)
    return Y


def R2(y_test, y_true):
    y_true = y_true.values
    return 1 - ((y_test - y_true)**2).sum() / (
        (y_true - y_true.mean())**2).sum()


# 绘图
def Show(x, y):
    x = np.array(x.T)
    y = np.array(y.T)
    line_x = np.linspace(0, 600000, 2)
    line_y = line_x
    plt.rcParams['font.family'] = ['STSong']
    plt.scatter(x, y, 10, c='r', label='预测值', marker='o')
    plt.plot(line_x, line_y, ls='-', lw=2, label='真实值', color='g')
    plt.legend(loc='best')
    plt.xlabel('真实房价', fontsize=15)
    plt.ylabel('预测房价', fontsize=15)
    plt.show()


# 读取数据
data = pd.read_csv('housing.csv')
data = data.dropna(axis=0, how='any')

# 将ocean_proximity列进行one hot encode处理
OneHot_data = pd.get_dummies(data[['ocean_proximity']])
data = data.join(OneHot_data)
data = data.drop(['ocean_proximity'], axis=1)

# 将房价中位数设为标签(median_house_value)，其余的设置为特征
# 划分训练集和测试集 
train_data = data.sample(frac=0.7, random_state=32, axis=0)# 取出训练集的X & Y
test_data = data[~data.index.isin(train_data.index)]  # 逆函数，剔除训练集

test_data_Y = test_data[['median_house_value']]  # 取出Y
test_data_X = test_data.drop(['median_house_value'], axis=1)  # 剔除Y
train_data_Y = train_data[['median_house_value']]  # 取出Y
train_data_X = train_data.drop(['median_house_value'], axis=1)  # 剔除Y

# 求闭式解,X为特征矩阵，y为标签
theta = close(train_data_X, train_data_Y)
test_data_Y_pred = pred(test_data_X, theta)  # 测试DataFrame→→→np.array

print('theta:')
print(theta)

# 对训练完的模型进行R2评估
print('R²:')
print(R2(test_data_Y_pred, test_data_Y))
Show(test_data_Y, test_data_Y_pred)
