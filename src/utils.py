from typing import Tuple, List, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def choose_worst_cases(y_true: np.ndarray, y_prob: np.ndarray, class_idx: int, k: int = 5):
    # False positives (prob high but y=0) and false negatives (prob low but y=1)
    prob = y_prob[:, class_idx]
    y = y_true[:, class_idx].astype(int)
    fps = np.where((y == 0) & (prob >= 0.5))[0]
    fns = np.where((y == 1) & (prob < 0.5))[0]
    # Rank by confidence margin
    fps_rank = fps[np.argsort(-prob[fps])]  # high prob first
    fns_rank = fns[np.argsort(prob[fns])]   # low prob first
    return fps_rank[:k], fns_rank[:k]

def basic_saliency(model: torch.nn.Module, img: torch.Tensor, class_idx: int) -> torch.Tensor:
    # img: (1,C,H,W) normalized
    model.eval()
    img = img.clone().requires_grad_(True)
    out = model(img)
    logit = out[0, class_idx]
    model.zero_grad(set_to_none=True)
    logit.backward(retain_graph=False)
    grad = img.grad.abs().sum(dim=1, keepdim=True)  # (1,1,H,W)
    # normalize
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
    return grad  # (1,1,H,W)

def overlay_saliency_on_image(img: torch.Tensor, sal: torch.Tensor):
    # img: (1,3,H,W) normalized; convert to [0,1] for display
    x = img[0].detach().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    s = sal[0,0].detach().cpu()
    fig = plt.figure()
    plt.imshow(x.permute(1,2,0))
    plt.imshow(s, alpha=0.35)  # simple overlay
    plt.axis('off')
    plt.tight_layout()
    return fig
