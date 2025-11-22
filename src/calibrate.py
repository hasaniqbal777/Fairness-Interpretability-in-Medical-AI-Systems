from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TemperatureScaler(nn.Module):
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1) + np.log(init_T))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

def fit_temperature(
    logits_val: np.ndarray,
    labels_val: np.ndarray,
    max_iter: int = 500,
    lr: float = 1e-2,
    verbose: bool = False,
) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemperatureScaler(init_T=1.0).to(device)
    logits = torch.tensor(logits_val, dtype=torch.float32, device=device)
    labels = torch.tensor(labels_val, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(max_iter):
        opt.zero_grad()
        loss = criterion(model(logits), labels)
        loss.backward()
        opt.step()
        if verbose and (i % 50 == 0):
            print(f"[TS] iter {i} NLL={loss.item():.4f} T={float(torch.exp(model.log_T)):.3f}")
    T = float(torch.exp(model.log_T).detach().cpu().item())
    return T

def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    return logits / T

def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # probs, labels: (N,) for binary — this function will be used per-class
    bins = np.linspace(0.0, 1.0, n_bins+1)
    mids = 0.5*(bins[1:] + bins[:-1])
    accs = np.zeros(n_bins, dtype=np.float32)
    confs = np.zeros(n_bins, dtype=np.float32)
    counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            accs[i] = labels[mask].mean()
            confs[i] = probs[mask].mean()
        else:
            accs[i] = np.nan
            confs[i] = np.nan
    return mids, accs, confs

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    N = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1])
        n = mask.sum()
        if n > 0:
            acc = labels[mask].mean()
            conf = probs[mask].mean()
            ece += (n / N) * abs(acc - conf)
    return float(ece)

def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(((probs - labels)**2).mean())

def plot_reliability(mids, accs, confs, title: str = "Reliability Diagram"):
    """Plot reliability diagram showing calibration.
    
    Args:
        mids: Bin midpoints (expected confidence levels)
        accs: Observed accuracy in each bin
        confs: Mean predicted confidence in each bin
        title: Plot title
    """
    plt.figure(figsize=(6,6))
    # Perfect calibration line
    plt.plot([0,1],[0,1], 'k--', label='Perfect calibration', alpha=0.5)
    
    # Valid points (bins with samples)
    mask = ~np.isnan(accs)
    
    # Plot actual calibration: bin midpoint (confidence) vs observed accuracy
    # Use confs (mean confidence) on x-axis for actual calibration visualization
    plt.plot(confs[mask], accs[mask], 'o-', markersize=8, linewidth=2, label='Model calibration')
    
    # Add gap indicators
    for i in np.where(mask)[0]:
        plt.plot([confs[i], confs[i]], [confs[i], accs[i]], 'r-', alpha=0.3, linewidth=1)
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence (Mean predicted probability)", fontsize=11)
    plt.ylabel("Accuracy (Observed frequency)", fontsize=11)
    plt.title(title, fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def risk_coverage_curve(y_true: np.ndarray, y_prob: np.ndarray, thresholds: Optional[List[float]] = None, n_points: int = 21):
    # Sort by uncertainty (1 - max prob for positive class; for binary use |p-0.5| as confidence)
    conf = np.abs(y_prob - 0.5)  # higher => more confident
    idx = np.argsort(conf)  # ascending (least confident first)
    coverages = []
    risks = []
    N = len(y_true)
    for k in range(n_points):
        frac = 1.0 - (k / (n_points-1))  # 1.0 -> 0.0
        # retain top CONFIDENT fraction
        take = int(np.round(frac * N))
        if take < 1:
            take = 1
        sel = idx[-take:]
        # classify with threshold 0.5 (can be tuned separately)
        y_hat = (y_prob[sel] >= 0.5).astype(int)
        err = (y_hat != y_true[sel]).mean() if take > 0 else 0.0
        coverages.append(frac)
        risks.append(err)
    return np.array(coverages), np.array(risks)
