import sys

import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.sparse import csgraph
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import pairwise_kernels


# def Lap_M_computing(K_V_V):  # 计算K(Vc,Vc)的拉普拉斯矩阵
#     D = np.diag(np.sum(K_V_V, axis=1))
#     L_D = D - K_V_V
#     d_temp = fractional_matrix_power(D, -0.5)
#     L_M = d_temp.dot(L_D).dot(d_temp)
#     return L_M


class HSIC_GHKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k_nn=2, lamda=0.5, gamma=0.5, beta=0.5, type='rbf', theta=1,
                 show_bar=False):
        '''

        :param k_nn: k NearestNeighbors for each class
        :param lamda:
        :param gamma:
        :param beta:
        :param type:
        '''
        self.k_nn = k_nn
        self.lamda = lamda
        self.gamma = gamma
        self.beta = beta
        self.type = type
        self.theta = theta
        self.show_bar = show_bar

    def fit(self, X, y, multi_views=[[1, 1], [2, 2]]):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        self.C_ = len(self.classes_)
        self.X_ = X
        self.y_ = y
        self.temp_dict_ = {
            "score_f": None,
            "test_X": None,
            "pred_y": None
        }
        self.multi_views_ = multi_views
        self.V_ = len(self.multi_views_)

        self.n_features_in_ = X.shape[1]
        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['classes_', 'C_', 'X_', 'y_', 'temp_dict_'])
        # Input validation
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError("the number of features in predict() is different from the number of features in fit")

        if (X == self.temp_dict_["test_X"]).all():
            return self.temp_dict_["pred_y"]
        n_test = X.shape[0]
        distance_s = np.zeros((n_test, self.C_))
        number_c = np.zeros((self.C_, 1))
        for i in range(self.C_):
            # print(np.argwhere(self.y == self.classes[i]).flatten())
            train_x_c = self.X_[np.argwhere(self.y_ == self.classes_[i]).flatten(), :]
            n_c = train_x_c.shape[0]
            number_c[i] = n_c

            if self.k_nn >= n_c:
                k_nn_i = n_c
            else:
                k_nn_i = self.k_nn
            # 调用sklearn的NearestNeighbors方法计算k近邻
            nbrs = NearestNeighbors(n_neighbors=k_nn_i, algorithm='auto').fit(train_x_c)
            indices = nbrs.kneighbors(X, return_distance=False)

            dim_k = k_nn_i
            matrics_H = np.eye(dim_k) - ((np.ones((dim_k, 1)).dot(np.ones((1, dim_k)))) / dim_k)

            round = 0
            for j in range(n_test):
                # 进度条显示
                if self.show_bar:
                    print("\r", end="")
                    prct = int(((j + 1) / n_test) * 100)
                    print("Class {}/{} , progress: {}/{}  {}%: ".format(i + 1, self.C_, j + 1, n_test, prct),
                          "▋" * (prct // 2), end="")
                    if j + 1 == n_test:
                        print("")
                    sys.stdout.flush()

                K_Vc_Vc_V = np.zeros((dim_k, dim_k, self.V_))  # k * k * V
                K_Vc_x_V = np.zeros((dim_k, 1, self.V_))  # k*1 * V
                L_M_V = np.zeros((dim_k, dim_k, self.V_))  # k * k * V
                K_x_x_V = np.zeros((1, self.V_))  # 1 * V
                alpha_V = np.zeros((dim_k, 1, self.V_))  # k*1 * V

                for v in range(self.V_):
                    # N个近邻样本
                    N_x_v = train_x_c[indices[j], (self.multi_views_[v][0] - 1):self.multi_views_[v][1]]
                    # 计算每一列的均值作为最终均值
                    N_mu_v = np.mean(N_x_v, axis=0)
                    Vc_v = N_x_v - N_mu_v  # 自动触发numpy广播
                    K_Vc_Vc_v = pairwise_kernels(Vc_v, Vc_v, metric=self.type, gamma=self.gamma, filter_params=True)
                    nc_x_v = (X[j, (self.multi_views_[v][0] - 1):self.multi_views_[v][1]] - N_mu_v).reshape(1, -1)
                    K_Vc_x_v = pairwise_kernels(Vc_v, nc_x_v, metric=self.type, gamma=self.gamma, filter_params=True)
                    L_M_v, d = csgraph.laplacian(K_Vc_Vc_v, return_diag=True, normed=True)
                    K_x_x_v = pairwise_kernels(nc_x_v, nc_x_v, metric=self.type, gamma=self.gamma, filter_params=True)

                    K_Vc_Vc_V[:, :, v] = K_Vc_Vc_v
                    K_Vc_x_V[:, :, v] = K_Vc_x_v
                    L_M_V[:, :, v] = L_M_v
                    K_x_x_V[:, v] = K_x_x_v
                for v in range(self.V_):  # 第v视角
                    alpha_v = np.linalg.inv(
                        K_Vc_Vc_V[:, :, v] + self.lamda * np.identity(dim_k) + self.beta * L_M_V[:, :, v]).dot(
                        K_Vc_x_V[:, :, v])
                    alpha_V[:, :, v] = alpha_v
                # 开始迭代
                round = 1
                while 1:
                    last_alpha_V = np.array(alpha_V)
                    matric_GAMMA = np.zeros((dim_k, dim_k))  # 辅助计算GAMMA_v的总和矩阵
                    matric_GAMMA_V = np.zeros((dim_k, dim_k, self.V_))  # 辅助计算GAMMA_v的单个矩阵
                    for v in range(self.V_):  # 第v视角
                        matric_GAMMA_v = matrics_H.dot(alpha_V[:, :, v]).dot(alpha_V[:, :, v].T).dot(
                            matrics_H)  # 未求和前的GAMMA矩阵每一项
                        matric_GAMMA_V[:, :, v] = matric_GAMMA_v  # 暂存以减少计算开销
                    matric_GAMMA = np.sum(matric_GAMMA_V, 2)  # 所有GAMMA矩阵每一项之和

                    # 更新alpha_V
                    for v in range(self.V_):
                        GAMMA_v = matric_GAMMA - matric_GAMMA_V[:, :, v]  # GAMMA_v, 由总和减去单个而得
                        alpha_v = np.linalg.inv(
                            K_Vc_Vc_V[:, :, v] + self.lamda * np.identity(dim_k) +
                            self.beta * L_M_V[:, :, v] + self.theta * GAMMA_v) \
                            .dot(K_Vc_x_V[:, :, v])
                        alpha_V[:, :, v] = alpha_v
                    if (round != 1 and np.linalg.norm(last_alpha_V - alpha_V) < 0.01 * np.linalg.norm(last_alpha_V)) \
                            or round >= 5:
                        #print(round)
                        break
                    round += 1

                dis_V = np.zeros((1, self.V_))
                for v in range(self.V_):
                    dis_V[:, v] = np.real(np.sqrt(K_x_x_V[:, v]
                                                  - 2 * K_Vc_x_V[:, :, v].T.dot(alpha_V[:, :, v])
                                                  + alpha_V[:, :, v].T.dot(K_Vc_Vc_V[:, :, v]).dot(alpha_V[:, :, v])))
                distance_s[j, i] = np.mean(dis_V)

                y_pred = self.classes_[distance_s.argmin(1)]
                sum_s = np.sum(distance_s, axis=1).reshape(-1, 1) - distance_s
                with np.errstate(divide='ignore', invalid='ignore'):
                    score_f = sum_s / np.sum(sum_s, axis=1).reshape(-1, 1)

                self.temp_dict_["score_f"] = score_f
                # noinspection PyTypedDict
                self.temp_dict_["test_X"] = X
                self.temp_dict_["pred_y"] = y_pred

        return y_pred

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self, ['classes_', 'C_', 'X_', 'y_', 'temp_dict_'])
        # Input validation
        X = check_array(X)
        if self.n_features_in_ != X.shape[1]:
            raise ValueError(
                "the number of features in predict_proba() is different from the number of features in fit")
        if (X == self.temp_dict_["test_X"]).all():
            return self.temp_dict_["score_f"]
        else:
            self.predict(X)
            return self.temp_dict_["score_f"]


if __name__ == "__main__":
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(HSIC_GHKNN())
