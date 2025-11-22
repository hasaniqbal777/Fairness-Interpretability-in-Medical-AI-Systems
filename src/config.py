from dataclasses import dataclass, field
from typing import List, Optional
import torch
import os
import random
import numpy as np

CHEST_XRAY14_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]

@dataclass
class Config:
    # Paths (edit these in the notebook)
    data_dir: str = "/path/to/ChestXray14/images"
    csv_path: str = "/path/to/ChestXray14/Data_Entry_2017.csv"
    train_val_list: Optional[str] = None  # e.g., "/.../train_val_list.txt"
    test_list: Optional[str] = None       # e.g., "/.../test_list.txt"
    checkpoint_path: str = "/path/to/model.pth"

    # Task setup
    target_classes: List[str] = field(default_factory=lambda: ["Pneumonia","Effusion","Atelectasis","Cardiomegaly"])
    primary_label: str = "Pneumonia"  # for risk/coverage & case studies

    # Data
    img_size: int = 320
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    # Eval and plots
    thresholds: Optional[List[float]] = None  # per-class thresholds (None => defaults to 0.5 or ROC-tuned in notebook)
    ece_bins: int = 15
    risk_cov_points: int = 21  # coverage grid points
    mc_dropout_passes: int = 10
    tta_passes: int = 8

    # Device/seed
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed: int = 1337

    # Normalization (ImageNet)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    def fix_seeds(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = False  # setting True may slow down and still not fully deterministic
        torch.backends.cudnn.benchmark = True

    def validate(self):
        for p in [self.data_dir, self.csv_path]:
            if p and not os.path.exists(p):
                print(f"[WARN] Path not found: {p}")
        if self.train_val_list and not os.path.exists(self.train_val_list):
            print(f"[WARN] train_val_list not found: {self.train_val_list}")
        if self.test_list and not os.path.exists(self.test_list):
            print(f"[WARN] test_list not found: {self.test_list}")
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            print(f"[WARN] checkpoint_path not found: {self.checkpoint_path}")
