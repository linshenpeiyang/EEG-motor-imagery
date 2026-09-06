"""升级包第四步：CSP（Common Spatial Patterns，共空间模式）。

动机：C3/C4 两个特征跨人上岗约 60%，是简单特征的天花板。
CSP 给 64 个通道学一组"空间滤波权重"，使两类信号的方差对比最大化：
一类信号通过滤波器后方差最大、另一类最小——
相当于把左右手想象的区别"调到最亮"再量。

数学直觉（不背公式版）：
  1. 每个类别算"平均协方差矩阵"，描述该类信号在通道之间怎么一起波动；
  2. 解广义特征值问题，找一组通道权重，让两类方差的比值最大化/最小化；
  3. 取最极端的 k 个权重当滤波器，特征 = 滤波后信号的 log 方差。
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(BaseEstimator, TransformerMixin):
    """fit 学滤波器权重；transform 输出每个滤波器的 log 方差特征。"""

    def __init__(self, n_components=6):
        self.n_components = n_components

    def fit(self, X, y):
        X = np.asarray(X)
        covs = []
        for c in np.unique(y):
            Xi = X[y == c]
            Xi = Xi - Xi.mean(axis=2, keepdims=True)  # 每个试次在时间上中心化
            cov = np.einsum("nct,nkt->nck", Xi, Xi).mean(axis=0)
            covs.append(cov)
        # 少量对角正则化，保证矩阵可逆、求解稳定
        covs[0] += np.eye(covs[0].shape[0]) * 1e-6
        covs[1] += np.eye(covs[1].shape[0]) * 1e-6
        # 广义特征值问题：eigh 返回升序，
        # 前 k 个让类 1 相对方差最大，后 k 个让类 0 相对方差最大
        _, evecs = eigh(covs[0], covs[0] + covs[1])
        k = self.n_components // 2
        self.filters_ = np.hstack([evecs[:, :k], evecs[:, -k:]])
        return self

    def transform(self, X):
        X = np.asarray(X)
        Z = np.einsum("nct,cf->nft", X, self.filters_)  # 空间滤波后的信号
        return np.log(Z.var(axis=2) + 1e-8)            # log 方差特征
