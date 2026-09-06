"""第 2 步：下载并加载 PhysioNet 运动想象数据。

做什么：
  1. 下载受试者 1 的三段"想象左右手握拳"实验（run 4/8/12）；
  2. 把三段连续记录拼接成一段；
  3. 通道名标准化、挂上标准头颅坐标（后面画脑地形图要用）；
  4. 把记录里的 T1（想象左手）/ T2（想象右手）标记解析成事件表。
"""
from pathlib import Path

import mne
from mne.datasets import eegbci

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mne_data"

SUBJECT = 1
RUNS = [4, 8, 12]  # 运动想象（左右手握拳）的三段重复实验

# 1) 定位 EDF 文件：第一次运行会联网下载，之后走本地缓存
#    update_path=True：数据下载后自动把缓存目录设为默认路径，避免交互式提问卡住脚本
raw_fnames = eegbci.load_data(SUBJECT, runs=RUNS, path=DATA_DIR,
                              update_path=False)

# 2) 逐段读入并拼接成一段连续记录
raw = mne.concatenate_raws([mne.io.read_raw_edf(f, preload=True) for f in raw_fnames])

# 3) 通道名标准化 + 挂上标准头颅坐标（后面画地形图依赖这一步）
eegbci.standardize(raw)
montage = mne.channels.make_standard_montage("standard_1005")
raw.set_montage(montage, on_missing="warn")

# 4) 把 T1/T2 标注解析成事件矩阵（每一行 = 一个试次的起点）
events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=2, T2=3))

print("=== raw 信息 ===")
print(f"{raw.info['sfreq']} Hz | {len(raw.ch_names)} 通道")
print("前 5 个事件（列：样本点编号、上一个事件编号、本事件编号）:")
print(events[:5])
print("事件编码:", event_id)
