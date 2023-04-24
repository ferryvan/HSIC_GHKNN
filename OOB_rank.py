from sklearn.ensemble import RandomForestClassifier
import pandas as pd


import numpy as np
import time
import scipy.io as scio
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours

import xgboost as xgb

from HSIC_GHKNN import HSIC_GHKNN


def scale_dataset(X_train, X_test):
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# 取数据集
dataset = scio.loadmat("./Dataset_vesicular_9787_1832.mat")
dataset_reAATP = scio.loadmat("./Dataset_vesicular_reAATP_9787_1832.mat")
# print(dataset)

# 420
AATP_9787 = dataset['AATP_9787']
AATP_1832 = dataset['AATP_1832']
# 150
GE_9787 = dataset['GE_9787']
GE_1832 = dataset['GE_1832']
# 200
NMBAC_9787 = dataset['NMBAC_9787']
NMBAC_1832 = dataset['NMBAC_1832']
# 100
PSSM_AB_9787 = dataset['PSSM_AB_9787']
PSSM_AB_1832 = dataset['PSSM_AB_1832']
# 1040
PSSM_DWT_9787 = dataset['PSSM_DWT_9787']
PSSM_DWT_1832 = dataset['PSSM_DWT_1832']
# 180
PSSM_PSE_9787 = dataset['PSSM_PSE_9787']
PSSM_PSE_1832 = dataset['PSSM_PSE_1832']
# 420
reAATP_9787 = dataset_reAATP['reAATP_9787']
reAATP_1832 = dataset_reAATP['reAATP_1832']

# 标签
label_9787 = dataset['label_9787']
label_1832 = dataset['label_1832']

# 拼接
data_train_X = np.concatenate((AATP_9787, GE_9787, NMBAC_9787, PSSM_AB_9787, PSSM_DWT_9787, PSSM_PSE_9787, reAATP_9787),
                              axis=1)
data_train_y = label_9787.ravel()
data_test_X = np.concatenate((AATP_1832, GE_1832, NMBAC_1832, PSSM_AB_1832, PSSM_DWT_1832, PSSM_PSE_1832, reAATP_1832),
                             axis=1)
data_test_y = label_1832.ravel()

data_train_X = np.nan_to_num(data_train_X)
data_train_y = np.nan_to_num(data_train_y)
data_test_X = np.nan_to_num(data_test_X)
data_test_y = np.nan_to_num(data_test_y)

# 不平衡处理
data_train_X, data_train_y = EditedNearestNeighbours().fit_resample(data_train_X, data_train_y)
print(data_train_X.shape)

# 归一化
data_train_X, data_test_X = scale_dataset(data_train_X, data_test_X)

# MVEC

multi_views = np.array([
    [1, 420],  # AATP
])
dim_views = []  # 每个视角的维数具体有哪些，从0开始
for v in range(len(multi_views)):
    dim_views.extend([i for i in range(multi_views[v][0] - 1, multi_views[v][1])])


# 只取这一部分的话，需要对多视角进行修改
for v in range(len(multi_views)):
    diff = multi_views[v][1]-multi_views[v][0]
    if v==0:
        multi_views[v][0] = 1
        multi_views[v][1] = multi_views[v][0] +diff
    else:
        multi_views[v][0] = multi_views[v-1][1]+1
        multi_views[v][1] = multi_views[v][0] +diff
data_train_X, data_test_X = data_train_X[:, dim_views], data_test_X[:, dim_views]

#data_train_X, data_test_X = scale_dataset(data_train_X, data_test_X)
# 载入数据集

# 获取特征和标签
X = data_train_X
y = data_train_y

print(X.shape)
# 初始化随机森林模型，n_estimators表示树的数量，random_state为了结果可重复
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 获取特征重要性
feature_importances = rf.feature_importances_

# 找出与类别标签最相关的特征
n_top_features = 20 # 取前5个关联度最高的特征
relevant_feature_indices = feature_importances.argsort()[::-1][:n_top_features]
relevant_feature_importances = feature_importances[relevant_feature_indices]

# 输出相关特征的序号和关联程度
cnt = 0
res = []

print('与类别标签为1的类别最相关的特征的序号:')
print('氨基酸标号1-20')
print('排序号:特征序号:含义:重要程度')
print(relevant_feature_indices)
for x in range(len(relevant_feature_indices)):
    id = relevant_feature_indices[x]
    if(id<=20):
        continue
    if 20<=id<=419:
        id -=20
        # TPC计算中有：TPM = reshape(TPMM,[],1);此时TPM是先列后行展开的
        i = id%20
        j = id//20
        meaning = f'TPC中的H{i+1},{j+1}\t'

    print(f'{x}\t{relevant_feature_indices[x]}\t:{meaning}\t\t:{relevant_feature_importances[x]}\t')
    cnt+=1
    if cnt<=20:
        res.append([relevant_feature_indices[x],i+1,j+1,relevant_feature_importances[x]])

print(res)
# res = [[121, 2, 6, 0.012686469528748119], [128, 9, 6, 0.009920893716771366], [243, 4, 12, 0.009637740109904981], [188, 9, 9, 0.009560194082944841], [123, 4, 6, 0.008647030041774986], [122, 3, 6, 0.008196000176014281], [151, 12, 7, 0.007485824764110094], [148, 9, 7, 0.007350365217867243], [408, 9, 20, 0.007043956818827305], [141, 2, 7, 0.006237546809409775], [131, 12, 6, 0.005666229083397596], [248, 9, 12, 0.005554288531244059], [154, 15, 7, 0.005455173107800632], [143, 4, 7, 0.005145381029377842], [84, 5, 4, 0.005011628162352175], [124, 5, 6, 0.00485177874883646], [144, 5, 7, 0.004787288578233887], [411, 12, 20, 0.004616275793477545], [134, 15, 6, 0.004613110093784378], [183, 4, 9, 0.004417886549550064]]


    
# print('与类别标签为1的类别最相关的特征的关联程度:')
# print(relevant_feature_importances)
# for i, feature_importance in enumerate(relevant_feature_importances):
#     print(f'{i+1}: {feature_importance}')

