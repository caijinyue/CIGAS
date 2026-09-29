#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# ========================== DeLong 检验（矢量化实现，稳定高效） ==========================
def delong_p_value(y_true, y_score_current, y_score_ref):
    y_true = np.asarray(y_true)
    y_score_current = np.asarray(y_score_current)
    y_score_ref = np.asarray(y_score_ref)

    if len(np.unique(y_true)) < 2:
        return np.nan

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)

    pos_cur = y_score_current[pos_mask]
    neg_cur = y_score_current[neg_mask]
    pos_ref = y_score_ref[pos_mask]
    neg_ref = y_score_ref[neg_mask]

    n_pos, n_neg = len(pos_cur), len(neg_cur)
    if n_pos < 2 or n_neg < 2:
        return np.nan

    # V10: 对每个负样本的排名分数
    V10_cur = np.mean(pos_cur[:, None] > neg_cur, axis=0) + 0.5 * np.mean(pos_cur[:, None] == neg_cur, axis=0)
    V10_ref = np.mean(pos_ref[:, None] > neg_ref, axis=0) + 0.5 * np.mean(pos_ref[:, None] == neg_ref, axis=0)

    # V01: 对每个正样本的排名分数
    V01_cur = np.mean(neg_cur[:, None] > pos_cur, axis=0) + 0.5 * np.mean(neg_cur[:, None] == pos_cur, axis=0)
    V01_ref = np.mean(neg_ref[:, None] > pos_ref, axis=0) + 0.5 * np.mean(neg_ref[:, None] == pos_ref, axis=0)

    var_cur = np.var(V10_cur, ddof=0) / n_neg + np.var(V01_cur, ddof=0) / n_pos
    var_ref = np.var(V10_ref, ddof=0) / n_neg + np.var(V01_ref, ddof=0) / n_pos
    cov = (np.cov(V10_cur, V10_ref, ddof=0)[0,1] / n_neg +
           np.cov(V01_cur, V01_ref, ddof=0)[0,1] / n_pos)

    var_diff = var_cur + var_ref - 2 * cov
    if var_diff <= 0:
        var_diff = 1e-20

    z = (roc_auc_score(y_true, y_score_current) - roc_auc_score(y_true, y_score_ref)) / np.sqrt(var_diff)
    p = 2 * norm.sf(abs(z))
    return p

# ========================== 单个数据集计算函数（供多进程调用） ==========================
def compute_metrics_for_dataset(args):
    mode, prefix, task, ref_task, df, ref_df, root, toggle_pred = args

    is_overall = (prefix == '')
    dataset_name = "Overall" if is_overall else prefix

    # 标签文件
    list0 = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "0_NGON"))]
    list1 = [os.path.splitext(f)[0] for f in os.listdir(os.path.join(root, "1_GON"))]

    true_labels, pred_probs, ref_probs = [], [], []
    class_1_prob_col = "class_0_prob" if toggle_pred else "class_1_prob"

    for _, row in df.iterrows():
        img_name = os.path.splitext(os.path.basename(row["image_path"]))[0]

        if not is_overall and not img_name.startswith(prefix + "-"):
            continue
        if img_name not in list0 and img_name not in list1:
            continue
        if ref_df is not None and img_name not in ref_df.index:
            continue  # 保证 DeLong 配对

        prob = row[class_1_prob_col]
        label = 0 if img_name in list0 else 1

        true_labels.append(label)
        pred_probs.append(prob)
        if ref_df is not None:
            ref_probs.append(ref_df.loc[img_name, class_1_prob_col])

    if len(true_labels) == 0:
        return None

    y_true = np.array(true_labels)
    y_prob = np.array(pred_probs)
    y_prob_ref = np.array(ref_probs) if ref_df is not None else None

    n_ngon = int((y_true == 0).sum())
    n_gon = int((y_true == 1).sum())

    # Youden Index 最佳阈值
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    best_thr = thresholds[best_idx]
    y_pred = (y_prob >= best_thr).astype(int)

    # 基础指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_prob)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred)

    # AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)

    # Bootstrap 1000 次（足够快了）
    rng = np.random.RandomState(42)
    boot = {'ACC':[], 'AUC':[], 'Sens':[], 'Spec':[], 'PPV':[], 'NPV':[], 'F1':[], 'AUPRC':[]}
    for _ in range(1000):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        y_p = (y_prob[idx] >= best_thr).astype(int)
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true[idx], y_p).ravel()

        boot['ACC'].append(accuracy_score(y_true[idx], y_p))
        boot['AUC'].append(roc_auc_score(y_true[idx], y_prob[idx]))
        boot['Sens'].append(tp_b/(tp_b+fn_b) if (tp_b+fn_b)>0 else 0)
        boot['Spec'].append(tn_b/(tn_b+fp_b) if (tn_b+fp_b)>0 else 0)
        boot['PPV'].append(tp_b/(tp_b+fp_b) if (tp_b+fp_b)>0 else 0)
        boot['NPV'].append(tn_b/(tn_b+fn_b) if (tn_b+fn_b)>0 else 0)
        boot['F1'].append(f1_score(y_true[idx], y_p))
        pr, rc, _ = precision_recall_curve(y_true[idx], y_prob[idx])
        boot['AUPRC'].append(auc(rc, pr))

    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boot.items()}

    # DeLong
    ref_auc_str = delong_p_str = "N/A"
    if y_prob_ref is not None:
        ref_auc = roc_auc_score(y_true, y_prob_ref)
        ref_auc_str = f"{ref_auc*100:.2f}"
        try:
            p = delong_p_value(y_true, y_prob, y_prob_ref)
            delong_p_str = f"{p:.016f}"
            # delong_p_str = "<0.001" if p < 0.001 else f"{p:.4f}"
        except:
            delong_p_str = "Error"

    return {
        "Mode": mode.upper(),
        "Dataset": dataset_name,
        "N": len(y_true),
        "NGON": n_ngon,
        "GON": n_gon,
        "Threshold": round(best_thr, 4),
        "ACC": f"{acc*100:.2f}",
        "ACC_CI": f"{ci['ACC'][0]*100:.2f}-{ci['ACC'][1]*100:.2f}",
        "AUC": f"{auc_roc*100:.2f}",
        "AUC_CI": f"{ci['AUC'][0]*100:.2f}-{ci['AUC'][1]*100:.2f}",
        "AUPRC": f"{auprc*100:.2f}",
        "AUPRC_CI": f"{ci['AUPRC'][0]*100:.2f}-{ci['AUPRC'][1]*100:.2f}",
        "Sensitivity": f"{sens*100:.2f}",
        "Sens_CI": f"{ci['Sens'][0]*100:.2f}-{ci['Sens'][1]*100:.2f}",
        "Specificity": f"{spec*100:.2f}",
        "Spec_CI": f"{ci['Spec'][0]*100:.2f}-{ci['Spec'][1]*100:.2f}",
        "PPV": f"{ppv*100:.2f}",
        "PPV_CI": f"{ci['PPV'][0]*100:.2f}-{ci['PPV'][1]*100:.2f}",
        "NPV": f"{npv*100:.2f}",
        "NPV_CI": f"{ci['NPV'][0]*100:.2f}-{ci['NPV'][1]*100:.2f}",
        "F1": f"{f1*100:.2f}",
        "F1_CI": f"{ci['F1'][0]*100:.2f}-{ci['F1'][1]*100:.2f}",
        "REF_AUC": ref_auc_str,
        "DeLong_p": delong_p_str
    }


def prediction_path(task, mode):
    directory = Path(os.environ.get("PREDICTIONS_ROOT", "outputs")) / task
    epoch = os.environ.get("EPOCH" if task != os.environ.get("REFERENCE_TASK", "") else "REFERENCE_EPOCH", "007")
    return str(directory / f"image_predictions_{mode}_epoch{int(epoch):03}.csv")

# ========================== 主程序 ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, help="你的新模型名")
    parser.add_argument("--toggle_pred", action="store_true", help="使用 class_0_prob 作为正类")
    args = parser.parse_args()

    task = args.task
    toggle_pred = args.toggle_pred
    ref_task = os.environ.get("REFERENCE_TASK", "")

    print(f"当前模型: {task}")
    print(f"对比模型: {ref_task}")
    print(f"toggle_pred: {toggle_pred}")

    modes = ["val", "test", "external", "REFUGE2"]
    image_path_col = "image_path"
    class_1_prob_col = "class_0_prob" if toggle_pred else "class_1_prob"
    final_save_path = f"./delong/{task}_youden_all_modes_summary_with_delong.csv"

    all_results = []

    for mode in modes:
        print(f"\n{'='*30} Processing {mode.upper()} {'='*30}")

        cur_path = prediction_path(task, mode)
        ref_path = prediction_path(ref_task, mode)

        if not os.path.exists(cur_path):
            print(f"未找到: {cur_path} → 跳过")
            continue

        df = pd.read_csv(cur_path)

        # 参考模型（可能不存在）
        ref_df = None
        if ref_task and os.path.exists(ref_path):
            ref_df = pd.read_csv(ref_path)
            ref_df['img_name'] = ref_df[image_path_col].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
            ref_df = ref_df.set_index('img_name')

        # 数据路径 & prefix 列表
        if mode == "external":
            root = os.path.join(os.environ["DATA_ROOT"], "SMDG", "test")
            prefixs = ['sjchoi86', 'BEH', 'G1020', 'PAPILA', 'CRFO', 'LES',
                       'HRF', 'OIA', 'JSIEC', 'REFUGE1', 'FIVES', 'ORIGA', 'DRISHTI', '']
        else:
            root = os.path.join(os.environ["DATA_ROOT"], "REFUGE2", "test") if mode == "REFUGE2" else os.path.join(os.environ["DATA_ROOT"], "DOVS", mode)
            prefixs = ['']

        # 构造多进程任务列表
        tasks = []
        for prefix in prefixs:
            tasks.append((mode, prefix, task, ref_task, df, ref_df, root, toggle_pred))

        # 多进程并行执行
        with ProcessPoolExecutor(max_workers=min(32, len(tasks))) as executor:
            futures = [executor.submit(compute_metrics_for_dataset, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{mode} 并行计算"):
                result = future.result()
                if result:
                    all_results.append(result)

    # ========================== 保存结果 ==========================
    if all_results:
        final_df = pd.DataFrame(all_results)
        final_df["Mode_order"] = final_df["Mode"].map({"VAL": 0, "TEST": 1, "EXTERNAL": 2})
        final_df["Is_Overall"] = (final_df["Dataset"] == "Overall").astype(int)
        final_df = final_df.sort_values(["Mode_order", "Is_Overall", "Dataset"],
                                        ascending=[True, False, True]).drop(columns=["Mode_order", "Is_Overall"])

        col_order = ["Mode","Dataset","N","NGON","GON","Threshold",
                     "ACC","ACC_CI","AUC","AUC_CI","REF_AUC","DeLong_p",
                     "AUPRC","AUPRC_CI",
                     "Sensitivity","Sens_CI","Specificity","Spec_CI",
                     "PPV","PPV_CI","NPV","NPV_CI","F1","F1_CI"]
        final_df = final_df[col_order]

        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        final_df.to_csv(final_save_path, index=False)

        print("\n" + "="*100)
        print("全部完成！结果已保存（含 AUPRC + DeLong p 值 + 多进程加速）")
        print(final_save_path)
        print("="*100)
        print(final_df.to_string(index=False))
    else:
        print("无结果生成！")

if __name__ == "__main__":
    main()
