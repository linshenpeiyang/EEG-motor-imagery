"""升级包第二步：把 10 位受试者的每段 run 切成一个"批次"。

核心思想：助手本体（analyze/compare/report/app）一行不用改——
它只认"批次文件"这个接口。本脚本把每位受试者的每段 run 做成一个
自包含的 npz 批次（信号+标签+通道名+采样率+受试者编号），
于是"跨 10 位受试者积累记忆"就变成"把 30 个批次喂给同一个助手"。

预处理与 03_epochs 保持一致：滤波 7-30Hz、tmin=-1/tmax=4、
基线校正、reject=None（枕部 alpha 是真信号不是伪迹）。
"""
from pathlib import Path

import numpy as np
import mne
from mne.datasets import eegbci

SUBJECTS = list(range(1, 11))
RUNS = [4, 8, 12]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "mne_data"
SUBJECT_DIR = PROJECT_ROOT / "data" / "subjects"


def make_run_batch(subject, run):
    """把一位受试者的一段 run 预处理成自包含批次文件。"""
    fname = eegbci.load_data(subject, runs=[run], path=DATA_DIR,
                             update_path=False)[0]
    raw = mne.io.read_raw_edf(fname, preload=True)
    eegbci.standardize(raw)
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="warn")
    raw.filter(7.0, 30.0, verbose=False)

    events, event_id = mne.events_from_annotations(raw,
                                                   event_id=dict(T1=2, T2=3))
    if len(events) == 0:
        return None
    epochs = mne.Epochs(raw, events, event_id, -1.0, 4.0,
                        baseline=(None, 0), reject=None, preload=True,
                        verbose=False)

    out = SUBJECT_DIR / f"S{subject:03d}R{run:02d}.npz"
    np.savez_compressed(out,
                        data=epochs.get_data(),
                        labels=epochs.events[:, -1] - 2,
                        ch_names=np.asarray(epochs.ch_names),
                        sfreq=epochs.info["sfreq"],
                        subject=subject, run=run)
    return out, len(epochs)


if __name__ == "__main__":
    SUBJECT_DIR.mkdir(exist_ok=True)
    made = []
    for s in SUBJECTS:
        for r in RUNS:
            res = make_run_batch(s, r)
            if res:
                out, n = res
                made.append(out)
                print(f"{out.name}: {n} 个试次")
    print(f"\n共生成 {len(made)} 个受试者批次文件"
          f"（受试者 {SUBJECTS[0]}~{SUBJECTS[-1]}）")
