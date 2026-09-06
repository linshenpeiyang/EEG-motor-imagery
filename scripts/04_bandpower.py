"""第 5-6 步：频带能量特征 + 逻辑回归基线。

特征设计（对应生理机制）：
  想象左手运动 -> 右侧运动皮层(C4)去同步 -> C4 的 mu/beta 能量下降；
  想象右手运动 -> 左侧运动皮层(C3)去同步 -> C3 能量下降。
所以每个试次只取两个数：C3、C4 在 8-30 Hz 的平均 log 能量，
用逻辑回归判断"哪一侧能量降得更明显"。
"""
from pathlib import Path

import numpy as np

import mne
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
epochs = mne.read_epochs(PROJECT_ROOT / "data" / "epochs-epo.fif",
                         preload=True)
X = epochs.get_data()                    # (试次, 64, 801)
y = epochs.events[:, -1] - 2             # 事件码 2/3 -> 标签 0/1

sfreq = epochs.info["sfreq"]
picks = mne.pick_channels(epochs.ch_names, ["C3", "C4"])

# 每个试次在 8-30 Hz 的功率谱，取 C3/C4 的平均能量并取对数
psds, freqs = psd_array_multitaper(X, sfreq=sfreq, fmin=8, fmax=30)
feat = np.log(psds[:, picks, :].mean(axis=-1))   # (45, 2)

# 先看生理效应：两类条件下 C3/C4 的 log 能量
print("各类别 C3/C4 平均 log 能量（越低=去同步越强）:")
for name, lab in [("左手想象", 0), ("右手想象", 1)]:
    mean = feat[y == lab].mean(axis=0)
    print(f"  {name}: C3={mean[0]:.3f}  C4={mean[1]:.3f}")

# 分类：标准化 + 逻辑回归，10 折交叉验证
clf = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(clf, feat, y, cv=10)
print(f"\n频带能量 + 逻辑回归 精度: {scores.mean():.3f} ± {scores.std():.3f}")
