import numpy as np
import time
import scipy.io as scio
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.under_sampling import EditedNearestNeighbours

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
# RandomUnderSampler(random_state=0) ,SMOTE(), RandomOverSampler(random_state=0)
data_train_X, data_train_y = EditedNearestNeighbours().fit_resample(data_train_X, data_train_y)

# 归一化
data_train_X, data_test_X = scale_dataset(data_train_X, data_test_X)

# MVEC

multi_views = np.array([
    [1, 420],  # AATP
    #[421, 570],  # GE
    #[571, 770],  # NMBAC
    #[771, 870],  # PSSM_AB
    #[871, 1910],  # PSSM_DWT
    [1911, 2090],  # PSSM_PSE
    #[2091, 2510],  # reAATP

])
dim_views = []  # 每个视角的维数具体有哪些，从0开始
for v in range(len(multi_views)):
    dim_views.extend([i for i in range(multi_views[v][0] - 1, multi_views[v][1])])

#data_train_X, data_test_X = data_train_X[:, dim_views], data_test_X[:, dim_views]
data_train_X, data_test_X = scale_dataset(data_train_X, data_test_X)

############################################################################

le = LabelEncoder()
data_train_y = le.fit_transform(data_train_y)


clf = HSIC_GHKNN(k_nn=650, lamda=0.1, gamma=0.4, beta=0.1, theta=5, type='rbf', show_bar=True)
clf.fit(data_train_X, data_train_y, multi_views)
#clf.fit(data_train_X, data_train_y)

y_pred = clf.predict(data_test_X)
y_pred = le.inverse_transform(y_pred)
y_score = clf.predict_proba(data_test_X)

tn, fp, fn, tp = confusion_matrix(data_test_y, y_pred).ravel()
metrics = {
    "SN": tp / (tp + fn),
    "PE": tp / (tp + fp),
    "SP": tn / (tn + fp),
    "ACC": accuracy_score(data_test_y, y_pred),
    "MCC": matthews_corrcoef(data_test_y, y_pred),
    "AUC": roc_auc_score(data_test_y, y_score[:, 1])
}
for i in metrics:
    metrics[i] = round(metrics[i], 4)

print(metrics)
