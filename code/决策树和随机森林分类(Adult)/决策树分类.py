import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import treelib as tb
from sklearn.metrics import roc_curve, auc
from treelib import node

# 建树
def Decision_Tree():
    classfiy=train_data.copy()
    Create_Tree(classfiy,root)

# 递归建树
def Create_Tree(classfiy,root):
    # 样本数量小于阈值，不再划分，取样本中分类结果为1的样本比例作为概率
    if len(classfiy)<pause_sample:
        try:
            root.data.column=classfiy['income'].value_counts(normalize=True).loc[1]
        except:root.data.column=0
        root.data.value=root.data.column
        root.data.str='概率：'+str(root.data.value)
        return
    
    try :
        # 如果树中的样本类别全为1，则无须划分
        if classfiy['income'].value_counts(normalize=True).loc[1]==1 :
            root.data.column=root.data.value=1
            root.data.str='概率：'+str(root.data.value)
            return

        # try报错，说明classfiy['income'].value_counts(normalize=True).loc[1]不存在
        # 即不存在类别为1的样本，则样本类型全为0，也无须划分
    except : 
            root.data.column=root.data.value=0
            root.data.str='概率：'+str(root.data.value)
            return
    
    # 更新当前节点的样本数
    root.data.sample=len(classfiy)
    
    # 存储每一列的基尼系数最小的种类及其基尼系数
    Gini_min=[]

    # 列循环寻找每一列中基尼系数最小的种类及其基尼系数，存储在Gini_min中
    for i in range(classfiy.shape[1]-1) :
        # 对数值列采用另一种处理方法处理
        if classfiy.iloc[:,i].dtype=='int64' :

            # 数值离散化成20个区间，取区间均值为种类
            unique=pd.cut(np.unique(classfiy.iloc[:,i].values),bins=20).categories
            unique=unique.mid

            Gini_column={}
            #对数值列获得的每一个种类计算基尼系数,结果存储在字典Gini_column中
            for j in range(len(unique)):
                classfiy_temp=classfiy[classfiy.iloc[:,i]>=unique[j]]
                classfiy_temp_N=classfiy[~classfiy.index.isin(classfiy_temp.index)]
                classfiy_temp_Y1=classfiy_temp[classfiy_temp.income==1]
                classfiy_temp_N_Y1=classfiy_temp_N[classfiy_temp_N.income==1]

                if len(classfiy_temp_N)==0 :
                    Gini=(len(classfiy_temp)/len(classfiy))*2*((len(classfiy_temp_Y1)/len(classfiy_temp))*(1-len(classfiy_temp_Y1)/len(classfiy_temp)))
                elif len(classfiy_temp)==0 :
                    Gini=(len(classfiy_temp_N)/len(classfiy))*2*((len(classfiy_temp_N_Y1)/len(classfiy_temp_N))*(1-len(classfiy_temp_N_Y1)/len(classfiy_temp_N)))
                else :
                    # 计算特征取某一值时的基尼系数
                    Gini=(len(classfiy_temp)/len(classfiy))*2*((len(classfiy_temp_Y1)/len(classfiy_temp))*(1-len(classfiy_temp_Y1)/len(classfiy_temp)))+(
                        len(classfiy_temp_N)/len(classfiy))*2*((len(classfiy_temp_N_Y1)/len(classfiy_temp_N))*(1-len(classfiy_temp_N_Y1)/len(classfiy_temp_N)))
                Gini_column[unique[j]]=Gini

            # 计算完一列的Gini数据后，找出最小值及其列名
            min_key=min(Gini_column,key=Gini_column.get)
            Gini_min.append({min_key:Gini_column[min_key]})

        # 处理非数值列
        else :
            # 取列的样本种类
            unique=classfiy.iloc[:,i].unique()
            Gini_column={}
            #对列的每一个种类计算基尼系数,结果存储在字典Gini_column中
            for j in range(unique.shape[0]):
                classfiy_temp=classfiy[classfiy.iloc[:,i]==unique[j]]
                classfiy_temp_N=classfiy[~classfiy.index.isin(classfiy_temp.index)]
                classfiy_temp_Y1=classfiy_temp[classfiy_temp.income==1]
                classfiy_temp_N_Y1=classfiy_temp_N[classfiy_temp_N.income==1]

                # 计算特征取某一值的基尼系数
                if len(classfiy_temp_N)==0 :
                    Gini=(len(classfiy_temp)/len(classfiy))*2*((len(classfiy_temp_Y1)/len(classfiy_temp))*(1-len(classfiy_temp_Y1)/len(classfiy_temp)))
                elif len(classfiy_temp)==0 :
                    Gini=(len(classfiy_temp_N)/len(classfiy))*2*((len(classfiy_temp_N_Y1)/len(classfiy_temp_N))*(1-len(classfiy_temp_N_Y1)/len(classfiy_temp_N)))
                else :
                    # 计算特征取某一值时的基尼系数
                    Gini=(len(classfiy_temp)/len(classfiy))*2*((len(classfiy_temp_Y1)/len(classfiy_temp))*(1-len(classfiy_temp_Y1)/len(classfiy_temp)))+(
                        len(classfiy_temp_N)/len(classfiy))*2*((len(classfiy_temp_N_Y1)/len(classfiy_temp_N))*(1-len(classfiy_temp_N_Y1)/len(classfiy_temp_N)))
                Gini_column[unique[j]]=Gini
                
            # 计算完一列的Gini数据后，找出最小值及其列名
            min_key=min(Gini_column,key=Gini_column.get)
            Gini_min.append({min_key:Gini_column[min_key]})
        
    # 列循环完毕，从Gini_min中找到最小值下标(下标即列索引)
    min_index=Gini_min.index(min(Gini_min,key=lambda x:list(x.values())[0]))

    # 更新当前节点信息
    root.data.column=classfiy.keys()[min_index]
    root.data.value=list(Gini_min[min_index])[0]
    root.data.gini=list((Gini_min)[min_index].values())[0]
    if type(root.data.value)==str:
        root.data.str=root.data.column+'=='+root.data.value
    else :root.data.str=root.data.column+'>='+str(root.data.value)
    # 根据基尼系数最小的特征值，划分数据集
    if type(list(Gini_min[min_index])[0])==str:
        classfiy1=classfiy[classfiy.iloc[:,min_index]==list(Gini_min[min_index])[0]]
        classfiy2=classfiy[~classfiy.index.isin(classfiy1.index)]
    else :
        classfiy1=classfiy[classfiy.iloc[:,min_index]>=list(Gini_min[min_index])[0]]
        classfiy2=classfiy[~classfiy.index.isin(classfiy1.index)]

    # 生成分支
    lchild=tree.create_node(parent=root.identifier,data=Node())
    rchild=tree.create_node(parent=root.identifier,data=Node())

    # 若基尼系数小于阈值，则结束递归，不再继续产生分支
    # 若当前节点深度达到界限,也结束递归
    if root.data.gini<pause_gini or tree.level(root.identifier)>=pause_depth:
        # 对最终划分出的样本，取样本中分类结果为1的样本比例作为概率
        if classfiy1.empty :
            try:
                rchild.data.value=classfiy2['income'].value_counts(normalize=True).loc[1]
            except:
                rchild.data.value=0
            rchild.data.column=rchild.data.value
            lchild.data.value=lchild.data.column=1-rchild.data.value

        elif classfiy2.empty :
            try:
                lchild.data.value=classfiy1['income'].value_counts(normalize=True).iloc[1]
            except:
                lchild.data.value=0
            lchild.data.column=lchild.data.value
            rchild.data.value=rchild.data.column=1-lchild.data.value
        
        else :
            try:
                lchild.data.value=lchild.data.column=classfiy1['income'].value_counts(normalize=True).loc[1]
            except:
                lchild.data.value=0
            try:
                rchild.data.value=rchild.data.column=classfiy2['income'].value_counts(normalize=True).loc[1]
            except:
                rchild.data.value=0
        lchild.data.str='概率：'+str(lchild.data.value)
        rchild.data.str='概率：'+str(rchild.data.value)
        return

    if classfiy1.empty==False :
        Create_Tree(classfiy1,lchild)
    if classfiy2.empty==False :
        Create_Tree(classfiy2,rchild)

# 保存树
def store(inputTree, filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()

# 读取树
def load(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)

# 预测结果
def pred(test_data_X):
    # 存储结果
    result=np.zeros(shape=[len(test_data_X),1])

    # 对测试集进行行迭代
    for i in range(len(test_data_X)):
        # 定义根节点
        node=root
        # 调用递归体
        pred_body(node,test_data_X.iloc[i],result,i)
    return result

# 使用递归根据决策树获取预测结果
def pred_body(node,data,result,i):
    # 叶节点，输出节点值(分类结果)
    if node.is_leaf() :
        result[i]=node.data.value

    # 若该节点存储的决策信息是字符串,启用字符串比对方法
    elif type(node.data.value)==str :
        # 根据coulumn存储的列信息获取对应列的值，比对
        if data[node.data.column]==node.data.value :
            # 若是当前节点是叶子节点，则直接输出结果
            node =tree.get_node(node.successors(tree_id=tree.identifier)[0])
            pred_body(node,data,result,i)
        else :
            node =tree.get_node(node.successors(tree_id=tree.identifier)[1])
            pred_body(node,data,result,i)
    
    #非字符串列，采用数值比对方法
    else :
        # 根据coulumn存储的列信息获取对应列的值，比对
        if data[node.data.column]>=node.data.value :
            # 若是当前节点是叶子节点，则直接输出结果
            node =tree.get_node(node.successors(tree_id=tree.identifier)[0])
            pred_body(node,data,result,i)
        else :
            node =tree.get_node(node.successors(tree_id=tree.identifier)[1])
            pred_body(node,data,result,i)

# 对预测概率的结果根据阈值进行分类处理
def classify(data_in,threshold):
    data_out=data_in.copy()
    for i in range(len(data_in)) :
        if data_in[i]>=threshold:
            data_out[i]=1
        else :data_out[i]=0
    return data_out.astype(np.int64)

# 统计准确性
def deviation(pred, true):
    pred = np.array(pred.T)
    true = np.array(true.T)
    count = 0
    for i in range(true.shape[1]):
        if true[0, i] == pred[0, i]:
            count = count + 1
    return count / true.shape[1]

# 画roc曲线
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


# 读取数据
data_source = pd.read_csv('adult.csv')

# 取X、Y
data_source_X = data_source.drop(labels=['income','educational-num'],axis=1)
data_source_Y = data_source[['income']]

# 取Y部分，把类别进行数值化处理
data_Y = data_source_Y
data_Y.income[data_Y.income == '>50K'] = 1
data_Y.income[data_Y.income == '<=50K'] = 0
data_Y['income'] = pd.to_numeric(data_Y['income'])

# 合成处理结果
data = data_source_X.join(data_Y)

# 取训练集和测试集
train_data = data.sample(frac=0.7, replace=False, axis=0, random_state=1)
test_data = data[~data.index.isin(train_data.index)]

# 取训练集的X、Y
train_data_X = train_data.drop(columns='income')
train_data_Y = train_data[['income']]

# 取测试集的X、Y
test_data_X = test_data.drop(columns='income')
test_data_Y = test_data[['income']]

# 节点
class Node(object):
    def __init__(self,lchild=None, rchild=None):
        self.column = ''
        self.value = ''
        self.sample=0
        self.gini=0
        self.str=''

# 决策树根节点
tree=tb.Tree()
root=tree.create_node('Root','root',data=Node())

# 设决策树的停止条件，一是深度界限，二是节点基尼系数界限,三是样本数（当节点样本数小于阈值，则不再分裂，取节点样本中类别为1的样本比例作为该叶子节点的概率）
# 限制树的深度
pause_depth=8
# 当节点基尼系数小于阈值，不再产生分支
pause_gini=0.06
# 样本数阈值
pause_sample=10 


# 建树
# 这里提供了，一组保存(store)和加载(load)树的函数
# 为节省反复运行时建树浪费的时间，可以调用这组函数保存和加载自己的树
#tree=load('tree')

root=tree.get_node('root')
# 构建决策树
Decision_Tree()

# 保存树
#store(tree,'tree')

# 预测,返回的数据是每个样本类型为1（>50K）的概率
pred_Y1_prob=pred(test_data_X)

# 设阈值
mythreshold=0.5

# 根据阈值分类
pred_Y=classify(pred_Y1_prob,mythreshold)

# 保存结果
#store(pred_Y,'Tree_pred')

# 准确率
print(deviation(pred_Y,test_data_Y.values))

# 计算真正率和假正率
fpr,tpr,threshold = roc_curve(test_data_Y, pred_Y1_prob)
#计算auc的值
roc_auc = auc(fpr,tpr)

#绘制roc曲线图
show_roc(fpr,tpr,roc_auc)
plt.show()