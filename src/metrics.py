from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve

def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')

def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float('nan')

def sensitivity_at_specificity(y_true: np.ndarray, y_score: np.ndarray, target_spec: float = 0.95) -> float:
    # Compute ROC curve and find threshold yielding specificity >= target_spec with max sensitivity
    fpr, tpr, thresh = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    mask = spec >= target_spec
    if mask.any():
        best_idx = np.argmax(tpr[mask])
        return float(tpr[mask][best_idx])
    # fallback: choose point with max specificity then report tpr
    best_idx = np.argmax(spec)
    return float(tpr[best_idx])

def f1_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.5) -> float:
    y_pred = (y_score >= thresh).astype(int)
    try:
        return float(f1_score(y_true, y_pred))
    except Exception:
        return float('nan')

def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresh: float = 0.5,
    spec_target: float = 0.95,
) -> Dict[str,float]:
    return {
        "AUROC": _safe_auroc(y_true, y_score),
        "AUPRC": _safe_auprc(y_true, y_score),
        "F1@{:.2f}".format(thresh): f1_at_threshold(y_true, y_score, thresh),
        "Sens@{:.0f}%Spec".format(spec_target*100): sensitivity_at_specificity(y_true, y_score, spec_target),
    }

def compute_multilabel_metrics(
    Y_true: np.ndarray,
    Y_score: np.ndarray,
    class_names: List[str],
    thresholds: Optional[List[float]] = None,
    spec_target: float = 0.95
) -> Dict[str,Dict[str,float]]:
    C = len(class_names)
    if thresholds is None:
        thresholds = [0.5]*C
    out = {}
    for i,c in enumerate(class_names):
        m = compute_binary_metrics(Y_true[:,i], Y_score[:,i], thresholds[i], spec_target)
        out[c] = m
    # Macro averages
    aurocs = [out[c]["AUROC"] for c in class_names if not np.isnan(out[c]["AUROC"])]
    auprcs = [out[c]["AUPRC"] for c in class_names if not np.isnan(out[c]["AUPRC"])]
    f1s   = [out[c]["F1@{:.2f}".format(thresholds[class_names.index(c)])] for c in class_names if not np.isnan(out[c]["AUROC"])]
    sens  = [out[c]["Sens@{:.0f}%Spec".format(spec_target*100)] for c in class_names if not np.isnan(out[c]["AUROC"])]
    out["MACRO"] = {
        "AUROC": float(np.nanmean(aurocs)) if len(aurocs)>0 else float('nan'),
        "AUPRC": float(np.nanmean(auprcs)) if len(auprcs)>0 else float('nan'),
        "F1": float(np.nanmean(f1s)) if len(f1s)>0 else float('nan'),
        "Sens@{:.0f}%Spec".format(spec_target*100): float(np.nanmean(sens)) if len(sens)>0 else float('nan'),
    }
    return out

def tune_threshold_for_f1(y_true: np.ndarray, y_score: np.ndarray, grid: np.ndarray = None) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1
    for t in grid:
        f1 = f1_at_threshold(y_true, y_score, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)
