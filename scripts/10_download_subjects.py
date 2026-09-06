"""升级包第一步：下载多位受试者的运动想象数据。

目标：把"一个人的 n"升级为"跨人的 n"。
把受试者 1~10 的三段运动想象 run（4/8/12）下载到项目本地 mne_data/，
后续脚本会把每位受试者切成批次，喂给同一个记忆助手（助手本体一行不改）。

磁盘预算：每位受试者约 3 个 EDF × 约 60 MB ≈ 180 MB，10 人试点约 1.8 GB；
验证流程后再按需扩展到全部 109 人。
"""
from pathlib import Path

from mne.datasets import eegbci

SUBJECTS = list(range(1, 11))      # 试点：受试者 1~10
RUNS = [4, 8, 12]                  # 运动想象（左右手握拳）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mne_data"


if __name__ == "__main__":
    files = eegbci.load_data(SUBJECTS, runs=RUNS, path=DATA_DIR,
                             update_path=False)
    print(f"共 {len(files)} 个 EDF 文件，已就绪（缓存于 {DATA_DIR}）")
    for f in files[:3]:
        print(" ", f)
    print("  ...")
