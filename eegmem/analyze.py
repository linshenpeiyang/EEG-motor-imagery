"""分析模块（Analyze）——从信号到"状态 + 置信度"（教学第 3 步）。

设计升级（第 3.5 课，修复上一轮发现的系统性偏左问题）：
  手写"谁低判谁"规则失效，是因为这位受试者的 C4 绝对能量普遍比 C3 低，
  规则把"通道固有偏移"误当成了"去同步差"。
  改为：用历史标定数据训练一个逻辑回归，作为助手上岗前的"先验知识"；
  上岗后模型冻结不动，新信号的"经验"交给记忆库积累。

诚实的数据切分：训练用 batch_01~04，上岗演示用 batch_05~09，
标定数据与上岗数据严格分开，避免"考试题就是练习题"的泄漏。
"""
from pathlib import Path

import joblib
import numpy as np
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "data" / "batches"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
CH_NAMES = ["C3", "C4"]
STATE_NAMES = {0: "左手想象", 1: "右手想象"}
CALIBRATION_BATCHES = ["batch_01", "batch_02", "batch_03", "batch_04"]


def extract_features(data, ch_names, sfreq, channels=CH_NAMES):
    """每个试次 -> 8-30 Hz 平均 log 能量。

    channels=None 时取全部 64 通道（给分类器用：判别力更稳）；
    默认 ["C3","C4"]（给记忆层用：可解释、可横向比较）。
    """
    picks = (slice(None) if channels is None
             else [list(ch_names).index(c) for c in channels])
    psds, _ = psd_array_multitaper(data, sfreq=sfreq, fmin=8, fmax=30,
                                   verbose=False)
    return np.log(psds[:, picks, :].mean(axis=-1))   # (试次, 通道数)


def session_zscore(feat):
    """会话内标准化：用本批自己的均值/标准差把特征归到同一尺度。

    跨受试者/跨会话时，绝对能量水平因人而异（电极位置、颅骨厚度、alpha 强弱），
    直接混合会让记忆库的"参考范围"失真（把正常值标成异常）。
    标准化后每批内部均值 0、标准差 1：保留"哪个通道相对被抑制得更狠"
    这一解码信息，去掉绝对水平差异。
    """
    return (feat - feat.mean(axis=0)) / feat.std(axis=0)


def make_model():
    """标准化 + 逻辑回归：和 04 步同款的基线模型。"""
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def train_model(feat, labels, model=None):
    """用标定数据训练先验模型。"""
    model = model or make_model()
    model.fit(feat, labels)
    return model


def classify(feat, model):
    """用冻结模型输出状态与置信度（预测类别的概率）。"""
    pred = model.predict(feat)
    prob = model.predict_proba(feat).max(axis=1)
    return pred.astype(int), prob


def analyze_batch(path, model):
    """用先验模型分析一个批次文件，返回逐试次结果列表。

    分类与记忆分离：分类器吃全通道特征（判别性），
    记忆库存 C3/C4（可比性）——同一批信号两套特征，各干各的。
    """
    with np.load(path) as z:
        feat_clf = session_zscore(extract_features(
            z["data"], z["ch_names"], float(z["sfreq"]), channels=None))
        feat_mem = session_zscore(extract_features(
            z["data"], z["ch_names"], float(z["sfreq"])))
        pred, prob = classify(feat_clf, model)
        return [
            {
                "trial": i,
                "c3": float(feat_mem[i, 0]),
                "c4": float(feat_mem[i, 1]),
                "pred": int(pred[i]),
                "prob": float(prob[i]),
                "label": int(z["labels"][i]),
            }
            for i in range(len(feat_mem))
        ]


def load_calibration(batch_names=CALIBRATION_BATCHES):
    """把标定批次的特征与标签拼起来，用于训练先验模型。"""
    feat, labels = [], []
    for name in batch_names:
        with np.load(BATCH_DIR / f"{name}.npz") as z:
            feat.append(session_zscore(
                extract_features(z["data"], z["ch_names"], float(z["sfreq"]),
                                 channels=None)))
            labels.append(z["labels"])
    return np.vstack(feat), np.concatenate(labels)


if __name__ == "__main__":
    # 标定：用前 4 批训练，交叉验证看它自己的水平，然后冻结保存
    X, y = load_calibration()
    model = train_model(X, y)
    scores = cross_val_score(make_model(), X, y, cv=5)
    print(f"先验模型在标定数据上的 5 折交叉验证精度: "
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    joblib.dump(model, MODEL_PATH)
    print(f"模型已冻结保存到 {MODEL_PATH.name}")

    # 上岗演示：用冻结模型分析第一个"新"批次 batch_05
    results = analyze_batch(BATCH_DIR / "batch_05.npz", model)
    print("\nbatch_05（模型从未见过的数据）:")
    print("试次 |   判定   | 置信度 |   真实   | 对错")
    for r in results:
        ok = "对" if r["pred"] == r["label"] else "错"
        print(f"  {r['trial']}  | {STATE_NAMES[r['pred']]} |  {r['prob']:.2f}  | "
              f"{STATE_NAMES[r['label']]} | {ok}")
    correct = sum(r["pred"] == r["label"] for r in results)
    print(f"本批 {len(results)} 个试次判对 {correct} 个 "
          f"({correct / len(results):.0%})")
