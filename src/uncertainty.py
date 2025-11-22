from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

def predictive_entropy(probs: np.ndarray) -> np.ndarray:
    # Binary/multi-label: entropy per sample per class, return mean entropy across classes
    eps = 1e-8
    ent = -(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    return ent.mean(axis=1)

@torch.no_grad()
def mc_dropout_passes(model: nn.Module, x: torch.Tensor, passes: int = 10) -> torch.Tensor:
    # Returns probs averaged over passes
    model.eval()
    # Enable dropout only
    for m in model.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d) or isinstance(m, nn.AlphaDropout):
            m.train()
    logits_list = []
    for _ in range(passes):
        logits = model(x)
        logits_list.append(torch.sigmoid(logits))
    probs = torch.stack(logits_list, dim=0).mean(dim=0)
    return probs

@torch.no_grad()
def tta_augment_and_predict(model: nn.Module, x: torch.Tensor, passes: int = 8) -> torch.Tensor:
    model.eval()
    # Simple TTA: horizontal flips and small intensity jitters
    logits_list = []
    for i in range(passes):
        aug = x.clone()
        if (i % 2) == 1:
            aug = torch.flip(aug, dims=[-1])
        # small jitter in normalized space
        aug = torch.clamp(aug + 0.02*torch.randn_like(aug), -3.0, 3.0)
        logits = model(aug)
        if (i % 2) == 1:
            logits = logits  # no need to unflip for classification logits
        logits_list.append(torch.sigmoid(logits))
    probs = torch.stack(logits_list, dim=0).mean(dim=0)
    return probs
