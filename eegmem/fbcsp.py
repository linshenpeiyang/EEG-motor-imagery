"""升级包第六步：FBCSP（Filter Bank CSP，滤波器组共空间模式）。

CSP 只在一个宽频带上找空间模式；FBCSP 把信号拆成多个子频带，
每个频带各自学 CSP，再把特征拼起来，并可用互信息挑选最有判别力的
"频带-滤波器"组合。它是运动想象领域的经典方法（Ang et al., 2008）。

注意：FBCSP 提升的是"判别力"，其空间模式仍然是个人的——
能否跨人迁移，交给实验回答，而不是想当然。
"""
import numpy as np
import scipy.signal
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from eegmem.csp import CSP

DEFAULT_BANDS = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]


def bandpass(X, band, fs=160.0):
    """4 阶 Butterworth 带通滤波（零相位）。"""
    sos = scipy.signal.butter(4, band, btype="bandpass", fs=fs,
                              output="sos")
    return scipy.signal.sosfiltfilt(sos, X, axis=2)


class FBCSP(BaseEstimator, TransformerMixin):
    """每个频带学一套 CSP 滤波器，拼特征，可选互信息筛选。"""

    def __init__(self, bands=DEFAULT_BANDS, n_components=2, select_k=None):
        self.bands = bands
        self.n_components = n_components
        self.select_k = select_k

    def fit(self, X, y):
        X = np.asarray(X)
        self.filters_per_band_ = []
        feats = []
        for band in self.bands:
            csp = CSP(n_components=self.n_components)
            feats.append(csp.fit_transform(bandpass(X, band), y))
            self.filters_per_band_.append(csp.filters_)
        self.all_features_ = np.hstack(feats)
        if self.select_k:
            self.sel_ = SelectKBest(mutual_info_classif,
                                    k=self.select_k).fit(self.all_features_, y)
        return self

    def transform(self, X):
        X = np.asarray(X)
        feats = []
        for band, W in zip(self.bands, self.filters_per_band_):
            Z = np.einsum("nct,cf->nft", bandpass(X, band), W)
            feats.append(np.log(Z.var(axis=2) + 1e-8))
        F = np.hstack(feats)
        if self.select_k:
            F = self.sel_.transform(F)
        return F
