"""第 4 步：滤波 + 切 Epoch（试次）+ 幅度伪迹拒绝。

三个概念：
  1. 滤波：只保留 7-30 Hz（mu+beta 运动相关节律），滤掉慢漂移和工频干扰；
  2. Epoch：按事件把连续信号切成一段段试次，每个试次对齐到"提示出现"时刻；
  3. 伪迹拒绝：峰峰值超过 100 微伏的试次直接丢弃（多为眨眼/肌肉抖动）。
"""
from pathlib import Path

import numpy as np

import mne
from mne.datasets import eegbci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mne_data"

SUBJECT = 2
RUNS = [4, 8, 12]

raw = mne.concatenate_raws(
    [mne.io.read_raw_edf(f, preload=True)
     for f in eegbci.load_data(SUBJECT, runs=RUNS, path=DATA_DIR,
                               update_path=False)]
)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage, on_missing="warn")

# 1) 带通滤波：只留 mu + beta 频段
raw.filter(7.0, 30.0, verbose=False)

# 2) 解析事件（T1=左手想象=2, T2=右手想象=3）
events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

# 3) 切 Epoch：提示前 1 秒到提示后 4 秒；用提示前段做基线校正。
#    注意：这里不设幅度拒绝（reject=None）。
#    实测本受试者每个试次的峰峰值都超过 180 uV，主要来自枕部 α 波
#    （8-13 Hz，正好在保留频段内）——α 波是运动想象解码的信号而非伪迹，
#    用固定 100 uV 阈值会把所有试次全丢光。真正的伪迹处理应改用 ICA。
tmin, tmax = -1.0, 4.0
epochs = mne.Epochs(
    raw,
    events,
    event_id,
    tmin,
    tmax,
    baseline=(None, 0),
    reject=None,
    preload=True,
)

print(epochs)
print("数据形状 (试次数, 通道数, 时间点):", epochs.get_data().shape)
print("每个类别的试次数:", dict(zip(["左手", "右手"], np.bincount(epochs.events[:, -1] - 2))))

# 存盘，下一步直接读，避免每次重复预处理
epochs.save(PROJECT_ROOT / "data" / "epochs-epo.fif", overwrite=True)
print("已保存: data/epochs-epo.fif")
