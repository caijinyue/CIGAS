import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, roc_curve
from sklearn.utils import resample
from scipy.stats import norm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import argparse

def delong_p_value(y_true, y_score_current, y_score_ref, auc_current=None, auc_ref=None):
    y_true = np.asarray(y_true)
    y_score_current = np.asarray(y_score_current)
    y_score_ref = np.asarray(y_score_ref)
    if len(np.unique(y_true)) < 2:
        return np.nan
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    pos_cur = y_score_current[pos_mask]
    neg_cur = y_score_current[neg_mask]
    pos_ref = y_score_ref[pos_mask]
    neg_ref = y_score_ref[neg_mask]
    n_pos, n_neg = (len(pos_cur), len(neg_cur))
    if n_pos < 2 or n_neg < 2:
        return np.nan
    V10_cur = np.mean(pos_cur[:, None] > neg_cur, axis=0) + 0.5 * np.mean(pos_cur[:, None] == neg_cur, axis=0)
    V10_ref = np.mean(pos_ref[:, None] > neg_ref, axis=0) + 0.5 * np.mean(pos_ref[:, None] == neg_ref, axis=0)
    V01_cur = np.mean(neg_cur[:, None] > pos_cur, axis=0) + 0.5 * np.mean(neg_cur[:, None] == pos_cur, axis=0)
    V01_ref = np.mean(neg_ref[:, None] > pos_ref, axis=0) + 0.5 * np.mean(neg_ref[:, None] == pos_ref, axis=0)
    var_cur = np.var(V10_cur, ddof=0) / n_neg + np.var(V01_cur, ddof=0) / n_pos
    var_ref = np.var(V10_ref, ddof=0) / n_neg + np.var(V01_ref, ddof=0) / n_pos
    cov = np.cov(V10_cur, V10_ref, ddof=0)[0, 1] / n_neg + np.cov(V01_cur, V01_ref, ddof=0)[0, 1] / n_pos
    var_diff = var_cur + var_ref - 2 * cov
    if var_diff <= 0:
        var_diff = 1e-20
    if auc_current is None:
        auc_current = roc_auc_score(y_true, y_score_current)
    if auc_ref is None:
        auc_ref = roc_auc_score(y_true, y_score_ref)
    z = (auc_current - auc_ref) / np.sqrt(var_diff)
    p = 2 * norm.sf(abs(z))
    return p

def compute_per_class_metrics(y_true, y_prob, y_prob_ref=None, class_name='Unknown'):
    n_samples = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n_samples - n_pos
    if n_pos == 0 or n_neg == 0:
        return {'Class': class_name, 'N': n_samples, 'N_Pos': n_pos, 'N_Neg': n_neg, 'Threshold': 'N/A', 'AUC': 'N/A', 'AUC_CI': 'N/A', 'AUPRC': 'N/A', 'AUPRC_CI': 'N/A', 'F1': 'N/A', 'F1_CI': 'N/A', 'Sensitivity': 'N/A', 'Sens_CI': 'N/A', 'Specificity': 'N/A', 'Spec_CI': 'N/A', 'PPV': 'N/A', 'PPV_CI': 'N/A', 'NPV': 'N/A', 'NPV_CI': 'N/A', 'ACC': 'N/A', 'ACC_CI': 'N/A', 'REF_AUC': 'N/A', 'DeLong_p': 'N/A'}
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    youden = tpr - fpr
    best_idx = np.argmax(youden)
    best_thr = thresholds[best_idx]
    y_pred = (y_prob >= best_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    auc_roc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp > 0 else 0
    npv = tn / (tn + fn) if tn + fn > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    f1 = f1_score(y_true, y_pred)
    n_boot = 500
    rng = np.random.RandomState(42)
    boot_indices = [rng.choice(n_samples, size=n_samples, replace=True) for _ in range(n_boot)]

    def _compute_boot_sample(idx):
        if len(np.unique(y_true[idx])) < 2 or y_true[idx].sum() == 0:
            return None
        y_p = (y_prob[idx] >= best_thr).astype(int)
        try:
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true[idx], y_p).ravel()
        except:
            return None
        return (roc_auc_score(y_true[idx], y_prob[idx]), average_precision_score(y_true[idx], y_prob[idx]), f1_score(y_true[idx], y_p), tp_b / (tp_b + fn_b) if tp_b + fn_b > 0 else 0, tn_b / (tn_b + fp_b) if tn_b + fp_b > 0 else 0, tp_b / (tp_b + fp_b) if tp_b + fp_b > 0 else 0, tn_b / (tn_b + fn_b) if tn_b + fn_b > 0 else 0, (tp_b + tn_b) / (tp_b + tn_b + fp_b + fn_b))
    boot_results = Parallel(n_jobs=-1, prefer='threads')((delayed(_compute_boot_sample)(idx) for idx in boot_indices))
    boot_results = [r for r in boot_results if r is not None]
    boot = {'AUC': [r[0] for r in boot_results], 'AUPRC': [r[1] for r in boot_results], 'F1': [r[2] for r in boot_results], 'Sens': [r[3] for r in boot_results], 'Spec': [r[4] for r in boot_results], 'PPV': [r[5] for r in boot_results], 'NPV': [r[6] for r in boot_results], 'ACC': [r[7] for r in boot_results]}
    ci = {k: np.percentile(v, [2.5, 97.5]) if v else ('N/A', 'N/A') for k, v in boot.items()}
    ref_auc_str = delong_p_str = 'N/A'
    if y_prob_ref is not None:
        try:
            ref_auc = roc_auc_score(y_true, y_prob_ref)
            ref_auc_str = f'{ref_auc * 100:.2f}'
            p = delong_p_value(y_true, y_prob, y_prob_ref, auc_current=auc_roc, auc_ref=ref_auc)
            delong_p_str = '<0.001' if p < 0.001 else f'{p:.4f}'
        except:
            delong_p_str = 'Error'
    return {'Class': class_name, 'N': n_samples, 'N_Pos': n_pos, 'N_Neg': n_neg, 'Threshold': round(best_thr, 4), 'AUC': f'{auc_roc * 100:.2f}', 'AUC_CI': f"{ci['AUC'][0] * 100:.2f}-{ci['AUC'][1] * 100:.2f}" if isinstance(ci['AUC'], np.ndarray) else 'N/A', 'Sensitivity': f'{sens * 100:.2f}', 'Sens_CI': f"{ci['Sens'][0] * 100:.2f}-{ci['Sens'][1] * 100:.2f}" if isinstance(ci['Sens'], np.ndarray) else 'N/A', 'Specificity': f'{spec * 100:.2f}', 'Spec_CI': f"{ci['Spec'][0] * 100:.2f}-{ci['Spec'][1] * 100:.2f}" if isinstance(ci['Spec'], np.ndarray) else 'N/A', 'PPV': f'{ppv * 100:.2f}', 'PPV_CI': f"{ci['PPV'][0] * 100:.2f}-{ci['PPV'][1] * 100:.2f}" if isinstance(ci['PPV'], np.ndarray) else 'N/A', 'NPV': f'{npv * 100:.2f}', 'NPV_CI': f"{ci['NPV'][0] * 100:.2f}-{ci['NPV'][1] * 100:.2f}" if isinstance(ci['NPV'], np.ndarray) else 'N/A', 'ACC': f'{acc * 100:.2f}', 'ACC_CI': f"{ci['ACC'][0] * 100:.2f}-{ci['ACC'][1] * 100:.2f}" if isinstance(ci['ACC'], np.ndarray) else 'N/A', 'F1': f'{f1 * 100:.2f}', 'F1_CI': f"{ci['F1'][0] * 100:.2f}-{ci['F1'][1] * 100:.2f}" if isinstance(ci['F1'], np.ndarray) else 'N/A', 'AUPRC': f'{auprc * 100:.2f}', 'AUPRC_CI': f"{ci['AUPRC'][0] * 100:.2f}-{ci['AUPRC'][1] * 100:.2f}" if isinstance(ci['AUPRC'], np.ndarray) else 'N/A', 'REF_AUC': ref_auc_str, 'DeLong_p': delong_p_str}

def _normalize_col_name(col):
    """Normalize column name: replace non-breaking spaces, collapse multiple spaces, strip."""
    import re
    col = col.replace('\xa0', ' ')
    col = re.sub('\\s+', ' ', col)
    return col.strip()

def load_and_merge_labels(mode, label_dir):
    """
    Load and merge GON and NGON label files for the given mode.
    Returns a DataFrame with 'img_name' and 12 unified label columns.
    """
    mode_file_map = {'val': ('val_GON_RY_20260105', 'val_NGON_RY20260211'), 'test': ('test_GON_RY_20260105', 'test_NGON_RY20260211'), 'external': ('SMDG_GON_RY_20260105', 'external_NGON_RY20260211'), 'REFUGE2': ('REFUGE2_GON_RY_20260105', 'REFUGE2_NGON_RY20260211')}
    if mode not in mode_file_map:
        print(f'Unknown mode for labels: {mode}')
        return None
    gon_stem, ngon_stem = mode_file_map[mode]

    def _read_file(stem):
        for ext in ['.csv', '.xlsx', '.xls']:
            fpath = os.path.join(label_dir, stem + ext)
            if os.path.exists(fpath):
                if ext == '.csv':
                    return pd.read_csv(fpath)
                else:
                    return pd.read_excel(fpath)
        print(f'File not found: {stem} in {label_dir}')
        return None
    df_gon = _read_file(gon_stem)
    df_ngon = _read_file(ngon_stem)
    if df_gon is None or df_ngon is None:
        return None
    df_gon.columns = [_normalize_col_name(c) for c in df_gon.columns]
    df_ngon.columns = [_normalize_col_name(c) for c in df_ngon.columns]
    unified_label_cols = ['enlarged vertical cup-to-disc ratio', 'glaucomatous disc cupping', 'laminar-dot sign within the cup', 'ISNT rule violation', 'neuro-retinal rim thinning/notching', 'beta-zone peripapillary atrophy', 'optic disc pallor relative to surrounding retina', 'nasalization of central retinal vessels', 'glaucomatous disc haemorrhages', 'vessel bayonetting at disc margin', 'wedge-shaped RNFL defect', 'macula & background retinal lesion']
    for col in unified_label_cols:
        if col not in df_gon.columns:
            df_gon[col] = 0
        if col not in df_ngon.columns:
            df_ngon[col] = 0
    if 'image' not in df_gon.columns or 'image' not in df_ngon.columns:
        print("Missing 'image' column in label files")
        return None
    df_gon = df_gon[['image'] + unified_label_cols].copy()
    df_ngon = df_ngon[['image'] + unified_label_cols].copy()
    df_merged = pd.concat([df_gon, df_ngon], ignore_index=True)
    df_merged['img_name'] = df_merged['image'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0].strip())
    return (df_merged, unified_label_cols)

def process_mode(args_tuple):
    mode, prefix, task, ref_task, severity_filter = args_tuple
    is_overall = prefix == ''
    dataset_name = 'Overall' if is_overall else prefix
    print(f'\nProcessing {mode.upper()} - {dataset_name}...')
    prediction_task = task
    epoch = int(os.environ.get('EPOCH', '10'))
    pred_path = os.path.join(os.environ.get('PREDICTIONS_ROOT', 'outputs'), prediction_task, f'image_predictions_{mode}_epoch{epoch:03}.csv')
    label_dir = os.path.join(os.environ['DATA_ROOT'], 'DOVS', 'label_multilabel_GON-NGON')
    if not os.path.exists(pred_path):
        print(f'未找到预测文件: {pred_path}')
        return []
    res = load_and_merge_labels(mode, label_dir)
    if res is None:
        return []
    df_true, label_cols = res
    df_true = df_true.set_index('img_name')
    df_pred = pd.read_csv(pred_path)
    df_pred['img_name'] = df_pred['image_path'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0].strip())
    df = df_pred.merge(df_true, on='img_name', how='left')
    df[label_cols] = df[label_cols].fillna(0).astype(int)
    if prefix:
        df = df[df['img_name'].str.startswith(str(prefix) + '-') | df['img_name'].str.startswith(str(prefix) + '_')]
    if len(df) == 0:
        print(f'{mode} - {dataset_name} 无数据')
        return []
    print(f'{mode} - {dataset_name} 总图像数: {len(df_pred)}, 过滤后: {len(df)}, 匹配到标签的图像: {df[label_cols].any(axis=1).sum()}')
    severity_matched_n = 0
    vf_path = os.path.join(os.environ['DATA_ROOT'], 'DOVS', 'DOVS_test_VF.csv')
    if os.path.exists(vf_path):
        df_vf = pd.read_csv(vf_path)
        df = df.merge(df_vf[['image_path', 'severity']], on='image_path', how='left')
        if severity_filter:
            severity_lower = severity_filter.lower()
            if severity_lower == 'severity':
                df = df[df['severity'].str.lower().str.contains('sev', na=False) | (df['severity'].str.lower() == 'severity') | df['severity'].isna()]
            elif severity_lower == 'mild':
                df = df[df['severity'].str.lower().str.contains('mild', na=False) | df['severity'].isna()]
            elif severity_lower == 'moderate':
                df = df[df['severity'].str.lower().str.contains('mod', na=False) | df['severity'].isna()]
        severity_matched_n = len(df)
    ref_prob = None
    prob_cols = [f'{col}_prob' for col in label_cols]
    ref_path = os.path.join(os.environ.get('PREDICTIONS_ROOT', 'outputs'), ref_task, f"image_predictions_{mode}_epoch{int(os.environ.get('REFERENCE_EPOCH', '10')):03}.csv")
    df_ref = None
    matched_ref = None
    if ref_task and os.path.exists(ref_path):
        df_ref = pd.read_csv(ref_path)
        df_ref['img_name'] = df_ref['image_path'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0].strip())
        df_ref = df_ref.set_index('img_name')
        if len(df_ref) > 0 and len(df) > 0:
            try:
                matched_ref = df_ref.reindex(df['img_name'])
            except Exception:
                pass
    results = []
    for col in label_cols:
        prob_col = f'{col}_prob'
        if prob_col not in df.columns:
            print(f'Warning: {prob_col} not found in predictions.')
            continue
        y_prob = df[prob_col].values.astype(float)
        y_true = df[col].values.astype(int)
        y_prob_ref = None
        if matched_ref is not None and prob_col in matched_ref.columns:
            try:
                y_prob_ref = matched_ref[prob_col].values
            except Exception:
                pass
        result = compute_per_class_metrics(y_true, y_prob, y_prob_ref, class_name=col)
        result['Mode'] = mode.upper()
        result['Dataset'] = dataset_name
        result['Severity_N'] = severity_matched_n
        results.append(result)
    if results:
        df_res = pd.DataFrame(results)
        total_n = df_res['N'].iloc[0] if len(df_res) > 0 else '-'
        valid_rows = df_res[pd.to_numeric(df_res['N_Pos'], errors='coerce') > 0]
        perf_cols = ['AUC', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'ACC', 'F1', 'AUPRC']
        if len(valid_rows) > 0:
            numeric_perf = valid_rows[perf_cols].replace({'N/A': np.nan, '-': np.nan})
            numeric_perf = numeric_perf.apply(pd.to_numeric, errors='coerce')
            mean_perf = numeric_perf.mean(skipna=True)
        else:
            mean_perf = pd.Series({col: np.nan for col in perf_cols})
        avg_row = {'Dataset': dataset_name, 'Class': 'Macro Avg', 'Mode': mode.upper(), 'N': total_n, 'N_Pos': '-', 'N_Neg': '-', 'Threshold': '-', 'AUC': f"{mean_perf['AUC']:.2f}" if pd.notna(mean_perf['AUC']) else '-', 'AUC_CI': '-', 'Sensitivity': f"{mean_perf['Sensitivity']:.2f}" if pd.notna(mean_perf['Sensitivity']) else '-', 'Sens_CI': '-', 'Specificity': f"{mean_perf['Specificity']:.2f}" if pd.notna(mean_perf['Specificity']) else '-', 'Spec_CI': '-', 'PPV': f"{mean_perf['PPV']:.2f}" if pd.notna(mean_perf['PPV']) else '-', 'PPV_CI': '-', 'NPV': f"{mean_perf['NPV']:.2f}" if pd.notna(mean_perf['NPV']) else '-', 'NPV_CI': '-', 'ACC': f"{mean_perf['ACC']:.2f}" if pd.notna(mean_perf['ACC']) else '-', 'ACC_CI': '-', 'F1': f"{mean_perf['F1']:.2f}" if pd.notna(mean_perf['F1']) else '-', 'F1_CI': '-', 'AUPRC': f"{mean_perf['AUPRC']:.2f}" if pd.notna(mean_perf['AUPRC']) else '-', 'AUPRC_CI': '-', 'REF_AUC': '-', 'DeLong_p': '-', 'Severity_N': severity_matched_n}
        for col in df_res.columns:
            if col not in avg_row:
                avg_row[col] = '-'
        results = [avg_row] + results
    return results

def main():
    parser = argparse.ArgumentParser(description='Multi-label 真实性能评估（使用真实疾病名）')
    parser.add_argument('task', type=str, help='当前模型任务名')
    parser.add_argument('--severity', action='store_true', help='指定时启用severity分组模式: mild / moderate / severity')
    args = parser.parse_args()
    task = args.task
    ref_task = os.environ.get('REFERENCE_TASK', '')
    modes = ['val', 'test', 'external', 'REFUGE2']
    if args.severity:
        modes = ['test']
        severity_levels = ['mild', 'moderate', 'severity']
    else:
        severity_levels = [None]
    for severity_filter in severity_levels:
        if severity_filter is None:
            print(f"\n{'=' * 50}")
            print(f'处理全部数据（无severity过滤）')
            print(f"{'=' * 50}")
            save_path = f'./delong/{task}_multilabel_real_performance.csv'
            desc_suffix = '全部'
        else:
            print(f"\n{'=' * 50}")
            print(f'处理 severity: {severity_filter}')
            print(f"{'=' * 50}")
            save_path = f'./delong/{task}_multilabel_real_performance_{severity_filter}.csv'
            desc_suffix = severity_filter
        all_results = []
        tasks = []
        for mode in modes:
            if mode == 'external':
                prefixs = ['sjchoi86', 'BEH', 'G1020', 'CRFO', 'PAPILA', 'LES', 'HRF', 'OIA', 'JSIEC', 'REFUGE1', 'FIVES', 'ORIGA', 'DRISHTI', '']
            else:
                prefixs = ['']
            for prefix in prefixs:
                tasks.append((mode, prefix, task, ref_task, severity_filter))
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_mode, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'进度 ({desc_suffix})'):
                result = future.result()
                all_results.extend(result)
        if all_results:
            final_df = pd.DataFrame(all_results)
            desired_order = ['Mode', 'Dataset', 'Class', 'N', 'N_Pos', 'N_Neg', 'Severity_N', 'Threshold', 'AUC', 'AUC_CI', 'Sensitivity', 'Sens_CI', 'Specificity', 'Spec_CI', 'PPV', 'PPV_CI', 'NPV', 'NPV_CI', 'ACC', 'ACC_CI', 'F1', 'F1_CI', 'AUPRC', 'AUPRC_CI', 'REF_AUC', 'DeLong_p']
            cols = [col for col in desired_order if col in final_df.columns] + [col for col in final_df.columns if col not in desired_order]
            final_df = final_df[cols]
            if 'Class' in final_df.columns:
                final_df['Sort_Key'] = final_df['Class'].apply(lambda x: 0 if x == 'Macro Avg' else 1)
                final_df = final_df.sort_values(['Mode', 'Dataset', 'Sort_Key', 'Class']).reset_index(drop=True)
                final_df = final_df.drop(columns=['Sort_Key'])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            final_df.to_csv(save_path, index=False)
            if severity_filter is None:
                print(f'\n评估完成！结果已保存: {save_path}')
            else:
                print(f'\n{severity_filter} 评估完成！结果已保存: {save_path}')
            print(final_df.to_string(index=False))
        elif severity_filter is None:
            print('无结果生成')
        else:
            print(f'{severity_filter} 无结果生成')
if __name__ == '__main__':
    main()
