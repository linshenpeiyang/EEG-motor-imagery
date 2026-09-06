"""第 3 步：可视化 —— 先"看见"脑电和运动相关节律。

输出两张图：
  1. figures/02_psd_all.png   全脑平均功率谱（找 mu/beta 峰的位置）
  2. figures/02_c3c4.png      C3 与 C4 的功率谱对比
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互模式：把图存成文件而不是弹窗

import mne
from mne.datasets import eegbci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mne_data"
FIG_DIR = PROJECT_ROOT / "figures"

SUBJECT = 1
RUNS = [4, 8, 12]

os.makedirs(FIG_DIR, exist_ok=True)

raw = mne.concatenate_raws(
    [mne.io.read_raw_edf(f, preload=True)
     for f in eegbci.load_data(SUBJECT, runs=RUNS, path=DATA_DIR,
                               update_path=False)]
)
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage, on_missing="warn")

# 图1：所有通道平均后的功率谱。x 轴是频率，y 轴是能量。
spec = raw.compute_psd(fmin=1, fmax=45)
fig = spec.plot(average=True, show=False)
fig.savefig(FIG_DIR / "02_psd_all.png", dpi=150)

# 图2：只取 C3（左半球）和 C4（右半球），对比两边能量。
picks = mne.pick_channels(raw.ch_names, ["C3", "C4"])
spec34 = raw.compute_psd(fmin=1, fmax=45, picks=picks)
fig2 = spec34.plot(show=False)
fig2.savefig(FIG_DIR / "02_c3c4.png", dpi=150)

print("已保存: figures/02_psd_all.png, figures/02_c3c4.png")
