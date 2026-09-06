"""升级包第七步：跨人泛化的严格对比（leave-2-subjects-out × 5 折）。

问题：哪个模型能"稳定跨人泛化"？单次 1~4 训练 / 5~10 测试的数字不稳。
协议：把 10 位受试者分成 5 组、每组 2 人，轮流留 2 人当测试、
其余 8 人训练，报告 5 折均值±标准差。

参赛模型：
  1. C3/C4 频带能量（生理先验，现役模型）
  2. 全通道频带能量（64 维，信息更多但更易过拟合）
  3. CSP（个人空间模式，对照）
  4. FBCSP（多频带 CSP）
"""
import sys
from pathlib import Path

import numpy as np
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegmem.analyze import extract_features
from eegmem.csp import CSP
from eegmem.fbcsp import FBCSP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_DIR = PROJECT_ROOT / "data" / "subjects"
FOLDS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
ALL_SUBJECTS = list(range(1, 11))


def paths_of(subjects):
    return sorted(p for p in SUBJECT_DIR.glob("S*.npz")
                  if int(p.stem[1:4]) in subjects)


def load_raw(subjects):
    X, y = [], []
    for p in paths_of(subjects):
        with np.load(p) as z:
            X.append(z["data"])
            y.append(z["labels"])
    return np.vstack(X), np.concatenate(y)


def session_z(x):
    std = np.where(x.std(axis=0) == 0, 1.0, x.std(axis=0))
    return (x - x.mean(axis=0)) / std


def band_feats(subjects, channels=None):
    """每批会话内标准化的 log 频带能量；channels=None 时取全部 64 通道。"""
    F, y = [], []
    for p in paths_of(subjects):
        with np.load(p) as z:
            if channels is None:
                psds, _ = psd_array_multitaper(
                    z["data"], sfreq=float(z["sfreq"]),
                    fmin=8, fmax=30, verbose=False)
                f = session_z(np.log(psds.mean(axis=-1)))
            else:
                f = session_z(extract_features(z["data"], z["ch_names"],
                                               float(z["sfreq"])))
            F.append(f)
            y.append(z["labels"])
    return np.vstack(F), np.concatenate(y)


def logreg():
    return LogisticRegression(max_iter=1000)


MODELS = {
    "C3/C4(2维)": lambda: make_pipeline(StandardScaler(), logreg()),
    "全通道(64维)": lambda: make_pipeline(StandardScaler(), logreg()),
    "CSP(6)": lambda: make_pipeline(CSP(n_components=6),
                                    StandardScaler(), logreg()),
    "FBCSP(10维选8)": lambda: make_pipeline(
        FBCSP(n_components=2, select_k=8), StandardScaler(), logreg()),
}


if __name__ == "__main__":
    scores = {name: [] for name in MODELS}
    for test_subs in FOLDS:
        train_subs = [s for s in ALL_SUBJECTS if s not in test_subs]
        Xtr_raw, ytr = load_raw(train_subs)
        Xte_raw, yte = load_raw(test_subs)
        Ftr_c, ytr_c = band_feats(train_subs, channels=["C3", "C4"])
        Fte_c, yte_c = band_feats(test_subs, channels=["C3", "C4"])
        Ftr_all, _ = band_feats(train_subs, channels=None)
        Fte_all, _ = band_feats(test_subs, channels=None)

        for name, make in MODELS.items():
            if name.startswith("C3/C4"):
                acc = make().fit(Ftr_c, ytr_c).score(Fte_c, yte_c)
            elif name.startswith("全通道"):
                acc = make().fit(Ftr_all, ytr).score(Fte_all, yte)
            else:
                acc = make().fit(Xtr_raw, ytr).score(Xte_raw, yte)
            scores[name].append(acc)

    print("跨人泛化（留2人×5折，均值±标准差）:")
    for name, accs in scores.items():
        a = np.array(accs)
        print(f"  {name:16s} {a.mean():.3f} ± {a.std():.3f}   {accs}")
