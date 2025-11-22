from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score, f1_score

@torch.no_grad()
def infer_logits(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cpu",
    progress: bool = True,
    batch_transform: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Infer logits from model on a dataloader."""
    model.eval()
    all_logits = []
    all_labels = []
    all_meta = []
    it = tqdm(loader, desc="Infer", leave=False) if progress else loader
    for batch in it:
        # Handle both 2-tuple and 3-tuple returns from dataloader
        if len(batch) == 3:
            imgs, ys, metas = batch
            all_meta.extend(metas)
        else:
            imgs, ys = batch
            
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        if batch_transform is not None:
            imgs = batch_transform(imgs)
        logits = model(imgs)
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(ys.detach().cpu().numpy())
        
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return logits, labels, all_meta

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid activation to numpy array."""
    return 1.0 / (1.0 + np.exp(-x))

def evaluate_multilabel(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    thresholds: Optional[List[float]] = None,
    spec_target: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """Evaluate multilabel classification metrics."""
    try:
        from .metrics import compute_multilabel_metrics
        probs = sigmoid(logits)
        return compute_multilabel_metrics(labels, probs, class_names, thresholds, spec_target)
    except ImportError:
        # Fallback if metrics module is not available
        probs = sigmoid(logits)
        results = {}
        for i, name in enumerate(class_names):
            try:
                if np.unique(labels[:, i]).size > 1:
                    auc = roc_auc_score(labels[:, i], probs[:, i])
                else:
                    auc = np.nan
            except Exception:
                auc = np.nan
            results[name] = {"auc": auc}
        return results

def evaluate(model, dataloader, criterion, device, desc="[Test]", disease_list=None):
    """
    Comprehensive evaluation function with AUC-ROC and F1 scores per class.
    
    Args:
        model: PyTorch model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to run evaluation on
        desc: Description for progress bar
        disease_list: List of disease names (optional)
        
    Returns:
        Tuple of (loss, avg_auc, avg_f1, auc_dict, f1_dict)
    """
    model.eval()
    running_loss = 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=desc, leave=True)

        for batch in progress_bar:
            # Handle both 2-tuple and 3-tuple returns from dataloader
            if len(batch) == 3:
                inputs, labels, _ = batch
            else:
                inputs, labels = batch
                
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.sigmoid(outputs)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    test_loss = running_loss / len(dataloader)

    num_classes = all_labels.shape[1]
    
    # Use provided disease list or create default one
    if disease_list is None:
        disease_list = [f"Class_{i}" for i in range(num_classes)]

    # Compute AUC for each class
    auc_scores = []
    for i in range(num_classes):
        try:
            if np.unique(all_labels[:, i]).size > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            else:
                auc = np.nan
        except Exception:
            auc = np.nan
        auc_scores.append(auc)
    
    avg_auc = np.nanmean(auc_scores)

    for i, disease in enumerate(disease_list):
        if not np.isnan(auc_scores[i]):
            print(f"{desc} {disease} AUC-ROC: {auc_scores[i]:.4f}")

    auc_dict = {disease_list[i]: auc_scores[i] for i in range(num_classes)}

    # Compute binary predictions for all classes
    preds_binary = (all_preds > 0.5).astype(int)

    # Per-class F1 scores
    f1_scores = []
    for i in range(num_classes):
        try:
            f1 = f1_score(all_labels[:, i], preds_binary[:, i])
        except Exception:
            f1 = np.nan
        f1_scores.append(f1)
    
    avg_f1 = np.nanmean(f1_scores)

    # Print per-class F1
    for i, disease in enumerate(disease_list):
        if not np.isnan(f1_scores[i]):
            print(f"{desc} {disease} F1 Score: {f1_scores[i]:.4f}")

    # Build F1 dictionary
    f1_dict = {disease_list[i]: f1_scores[i] for i in range(num_classes)}
    print(f"{desc} Loss: {test_loss:.4f}, Avg AUC-ROC: {avg_auc:.4f}, Avg F1 Score: {avg_f1:.4f}")

    return test_loss, avg_auc, avg_f1, auc_dict, f1_dict
