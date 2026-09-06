"""批次接入（Batch Stream）——模拟"新检测信号一批批到达"（教学第 2 步）。

现实中的助手面对的是持续到达的信号流，而不是一次性把 45 个试次全塞进来。
这里把 epochs-epo.fif 的 45 个试次按【到达顺序】切成若干批次（每批 5 个），
每批保存成独立文件 batch_XX.npz，模拟"又有一批新信号进来了"。

两个关键点：
  1. 必须保持原始顺序，不能随机打乱——记忆系统要知道"事情发生的先后"，
     后面做趋势分析（状态随时间的漂移）全靠这个顺序；
  2. 每批带标签是为了教学阶段能核对对错；真实部署中新到的信号往往没有标签；
  3. 批次文件要"自包含"：信号、标签、通道名、采样率一起打包，
     后面任何一步只拿这一个文件就能分析，不用回头翻原始 epochs。
"""
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_SIZE = 5                     # 每批试次数：45 = 9 批 × 5 个，刚好整除
BATCH_DIR = PROJECT_ROOT / "data" / "batches"


def make_batches(epochs, batch_size=BATCH_SIZE, out_dir=BATCH_DIR):
    """按顺序切批，每批打包成一个 .npz 文件。返回所有批次文件路径。"""
    out_dir.mkdir(exist_ok=True)
    data = epochs.get_data()               # (45, 64, 801)
    labels = epochs.events[:, -1] - 2      # 事件码 2/3 -> 标签 0/1
    ch_names = np.asarray(epochs.ch_names)  # 64 个通道名（后面要找到 C3/C4）
    sfreq = epochs.info["sfreq"]            # 采样率（算功率谱要用）
    files = []
    for i in range(0, len(epochs), batch_size):
        batch_id = i // batch_size + 1
        path = out_dir / f"batch_{batch_id:02d}.npz"
        np.savez_compressed(path,
                            data=data[i:i + batch_size],
                            labels=labels[i:i + batch_size],
                            ch_names=ch_names,
                            sfreq=sfreq)
        files.append(path)
    return files


if __name__ == "__main__":
    epochs = mne.read_epochs(PROJECT_ROOT / "data" / "epochs-epo.fif",
                             preload=True)
    files = make_batches(epochs)
    print(f"共切成 {len(files)} 个批次，每批最多 {BATCH_SIZE} 个试次：")
    for path in files:
        z = np.load(path)
        print(f"  {path.name}: 试次 {z['data'].shape[0]} 个, "
              f"标签 {z['labels'].tolist()} (0=左手 1=右手)")
