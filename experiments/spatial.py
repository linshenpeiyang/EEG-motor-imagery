"""Original pilot CSP/FBCSP baselines with deterministic feature selection.

The absolute ridge and log offset preserve the original experiment. They depend
on signal scale and are recorded explicitly rather than presented as tuned defaults.
"""
from functools import partial

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

from eegmem.analyze import make_model

BANDS = ((8, 12), (12, 16), (16, 20), (20, 24), (24, 30))


class CSP(TransformerMixin, BaseEstimator):
    """Binary spatial filters selected from both ends of a generalized eigensystem."""

    def __init__(self, n_components=6, ridge=1e-6, log_offset=1e-8):
        self.n_components = n_components
        self.ridge = ridge
        self.log_offset = log_offset

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3 or y.shape != (len(X),) or len(np.unique(y)) != 2:
            raise ValueError('CSP requires trial/channel/time data and two label classes')
        if self.n_components < 2 or self.n_components % 2 or self.n_components > X.shape[1]:
            raise ValueError('CSP components must be even and between 2 and the channel count')
        covariances = []
        for label in np.unique(y):
            trials = X[y == label]
            centered = trials - trials.mean(axis=2, keepdims=True)
            covariance = np.einsum('nct,nkt->nck', centered, centered).mean(axis=0)
            covariances.append(covariance + np.eye(X.shape[1]) * self.ridge)
        _, vectors = eigh(covariances[0], covariances[0] + covariances[1])
        half = self.n_components // 2
        self.filters_ = np.hstack([vectors[:, :half], vectors[:, -half:]])
        return self

    def transform(self, X):
        check_is_fitted(self, 'filters_')
        projected = np.einsum('nct,cf->nft', X, self.filters_)
        return np.log(projected.var(axis=2) + self.log_offset)


class FBCSP(TransformerMixin, BaseEstimator):
    """Five sub-bands, two CSP features per band, training-only mutual information."""

    def __init__(self, sfreq=160., n_components=2, select_k=8, random_state=0):
        self.sfreq = sfreq
        self.n_components = n_components
        self.select_k = select_k
        self.random_state = random_state

    def fit(self, X, y):
        self.banks_ = []
        features = []
        for band in BANDS:
            sos = butter(4, band, btype='bandpass', fs=self.sfreq, output='sos')
            csp = CSP(n_components=self.n_components)
            features.append(csp.fit_transform(sosfiltfilt(sos, X, axis=2), y))
            self.banks_.append((sos, csp))
        self.selector_ = SelectKBest(
            partial(mutual_info_classif, random_state=self.random_state),
            k=self.select_k).fit(np.hstack(features), y)
        return self

    def transform(self, X):
        check_is_fitted(self, 'selector_')
        features = [csp.transform(sosfiltfilt(sos, X, axis=2)) for sos, csp in self.banks_]
        return self.selector_.transform(np.hstack(features))


def spatial_models(sfreq):
    """Use the same scaler and logistic regression as the spectral baselines."""
    return {'CSP': make_pipeline(CSP(), make_model()),
            'FBCSP': make_pipeline(FBCSP(sfreq=sfreq), make_model())}
