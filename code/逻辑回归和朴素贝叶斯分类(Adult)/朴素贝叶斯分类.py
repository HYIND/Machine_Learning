import pandas as pd
import numpy as np
import scipy.stats
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt

# 对于非数值列
# 统计P(Xi|Y=1)存储在prob_list第一行
# 统计P(Xi|Y=0)存储在prob_list第二行
def Probability(train_data):
    train_data_temp1=train_data_Y1.drop(labels=['age','fnlwgt','capital-gain','capital-loss','hours-per-week'],axis=1)
    train_data_temp2=train_data_Y0.drop(labels=['age','fnlwgt','capital-gain','capital-loss','hours-per-week'],axis=1)
    prob_list=np.zeros(shape=[2,98]).astype(np.float64)

    # 统计P(Xi|Y=1)存储在prob_list第一行
    for i in range (train_data_temp1.shape[1]-1):
        prob_list[0,i]=((len(train_data_temp1)-train_data_temp1.iloc[:,i].value_counts()[0])+1)/(len(train_data_temp1)+103)

    # 统计P(Xi|Y=0)存储在prob_list第二行
    for i in range (train_data_temp2.shape[1]-1):
        prob_list[1,i]=((len(train_data_temp2)-train_data_temp2.iloc[:,i].value_counts()[0])+1)/(len(train_data_temp2)+103)

    return prob_list

# 预测概率
def pred(test_data_X):
    # 拷贝
    test_data=test_data_X.copy()

    # 用于存储结果
    prob_pred=np.zeros(shape=[1,len(test_data)]).astype(np.float64)

    # 切片，取数值列(前五列)，用于存储数值部分概率P(X|Y=0),在下面会进行处理
    test_data_clip=test_data.iloc[:,:5]

    # 将数值部分转化为高斯(正态)分布下的概率P(X|Y=1)
    test_data['age']=scipy.stats.norm(mu_age_Y1, std_age_Y1).pdf(test_data['age'])
    test_data['fnlwgt']=scipy.stats.norm(mu_fnlwgt_Y1, std_fnlwgt_Y1).pdf(test_data['fnlwgt'])
    test_data['capital-gain']=scipy.stats.norm(mu_capital_gain_Y1, std_capital_gain_Y1).pdf(test_data['capital-gain'])
    test_data['capital-loss']=scipy.stats.norm(mu_capital_loss_Y1, std_capital_loss_Y1).pdf(test_data['capital-loss'])
    test_data['hours-per-week']=scipy.stats.norm(mu_housr_per_week_Y1, std_housr_per_week_Y1).pdf(test_data['hours-per-week'])

    # 同上，使用高峰高斯(正态)分布获得数值部分概率P(X|Y=0)
    test_data_clip['age']=scipy.stats.norm(mu_age_Y0, std_age_Y0).pdf(test_data_clip['age'])
    test_data_clip['fnlwgt']=scipy.stats.norm(mu_fnlwgt_Y0, std_fnlwgt_Y0).pdf(test_data_clip['fnlwgt'])
    test_data_clip['capital-gain']=scipy.stats.norm(mu_capital_gain_Y0, std_capital_gain_Y0).pdf(test_data_clip['capital-gain'])
    test_data_clip['capital-loss']=scipy.stats.norm(mu_capital_loss_Y0, std_capital_loss_Y0).pdf(test_data_clip['capital-loss'])
    test_data_clip['hours-per-week']=scipy.stats.norm(mu_housr_per_week_Y0, std_housr_per_week_Y0).pdf(test_data_clip['hours-per-week'])

    test_data=np.array(test_data)
    test_data_clip=np.array(test_data_clip)
    # 根据prob_list中的概率连乘,同时计算 ∏P(X|Y) 和 ∏P(X)
    for i in range (len(test_data)):
        # 先把当前行前5列相乘，也就是上述高斯分布预测的5个概率相乘
        # P(X|Y=1)
        P_x_y1_cumprod=test_data[i,0]*test_data[i,1]*test_data[i,2]*test_data[i,3]*test_data[i,4]
        # P(X|Y=0)
        P_x_y0_cumprod=test_data_clip[i,0]*test_data_clip[i,1]*test_data_clip[i,2]*test_data_clip[i,3]*test_data_clip[i,4]

        # 处理其他非数值列，概率取值于prob_list
        for j in range (5,test_data.shape[1]):
            if(test_data[i,j]==1) :
                P_x_y1_cumprod= P_x_y1_cumprod*prob_list[0,j-5]
                P_x_y0_cumprod=P_x_y0_cumprod*prob_list[1,j-5]
            else :
                P_x_y1_cumprod= P_x_y1_cumprod*(1-prob_list[0,j-5])
                P_x_y0_cumprod=P_x_y0_cumprod*(1-prob_list[1,j-5])

        prob_pred[0,i]=(P_Y1* P_x_y1_cumprod)/(P_Y1*P_x_y1_cumprod+P_Y0*P_x_y0_cumprod)
    
    return prob_pred

#  对预测概率的结果根据阈值进行分类处理
def classify(pred_sigmoid,threshold):
    data_out=pred_sigmoid.copy()
    for i in range(pred_sigmoid.shape[0]):
        inx=pred_sigmoid[i,0]
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
    plt.scatter(count, true, 10, c='r', label='真实值', marker='o')
    plt.scatter(count, pred, 10, c='g', label='预测值', marker='o')
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


data_source = pd.read_csv('adult.csv')

# 取X、Y
data_source_X = data_source.drop(labels=['income','educational-num'],axis=1)
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

# 取测试集的X、Y
test_data_X = test_data.drop(columns='income')
test_data_Y = test_data[['income']]

# 统计数值列的平均数和标准差，用于获得数值列的高斯分布(正态分布)函数，据此统计数值列的概率P(X|Y=1)
train_data_Y1=train_data[(train_data['income']==1)]
mu_age_Y1 = np.mean(train_data_Y1['age'])
std_age_Y1 = np.std(train_data_Y1['age'])
mu_fnlwgt_Y1=np.mean(train_data_Y1['fnlwgt'])
std_fnlwgt_Y1=np.std(train_data_Y1['fnlwgt'])
mu_capital_gain_Y1 = np.mean(train_data_Y1['capital-gain'])
std_capital_gain_Y1 = np.std(train_data_Y1['capital-gain'])
mu_capital_loss_Y1 = np.mean(train_data_Y1['capital-loss'])
std_capital_loss_Y1 = np.std(train_data_Y1['capital-loss'])
mu_housr_per_week_Y1 = np.mean(train_data_Y1['hours-per-week'])
std_housr_per_week_Y1 = np.std(train_data_Y1['hours-per-week'])

# 同上，取Y=0的列，用于获得数值列中Y=0部分的高斯分布(正态分布)函数，据此统计数值列的概率P(X|Y=0)
train_data_Y0=train_data[(train_data['income']==0)]
mu_age_Y0 = np.mean(train_data_Y0['age'])
std_age_Y0 = np.std(train_data_Y0['age'])
mu_fnlwgt_Y0=np.mean(train_data_Y0['fnlwgt'])
std_fnlwgt_Y0=np.std(train_data_Y0['fnlwgt'])
mu_capital_gain_Y0 = np.mean(train_data_Y0['capital-gain'])
std_capital_gain_Y0 = np.std(train_data_Y0['capital-gain'])
mu_capital_loss_Y0 = np.mean(train_data_Y0['capital-loss'])
std_capital_loss_Y0 = np.std(train_data_Y0['capital-loss'])
mu_housr_per_week_Y0 = np.mean(train_data_Y0['hours-per-week'])
std_housr_per_week_Y0 = np.std(train_data_Y0['hours-per-week'])

# 统计P(Xi|Y=1)和P(Xi|Y=0)
prob_list=Probability(train_data)

# 求P(Y=1)和P(Y=0)
P_Y1=np.float64(len(train_data[(train_data['income']==1)]))/np.float(len(train_data))
P_Y0=np.float64(len(train_data[(train_data['income']==0)]))/np.float64(len(train_data))

# 预测概率
prob_pred=pred(test_data_X)

# 将预测概率的结果按阈值分类
mythreshold=0.5
test_data_pred_classify=classify(prob_pred.T,mythreshold)

# 统计准确率
print(deviation(test_data_pred_classify, test_data_Y))

#绘制预测图
show(test_data_pred_classify, test_data_Y)

# 计算真正率和假正率
fpr,tpr,threshold = roc_curve(test_data_Y, prob_pred.T)
#计算auc的值
roc_auc = auc(fpr,tpr)

#绘制roc曲线图
show_roc(fpr,tpr,roc_auc)

#显示图
plt.show()
