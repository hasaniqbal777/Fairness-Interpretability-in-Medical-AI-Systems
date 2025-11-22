# CV8502 — Fairness & Explainability Analysis of DenseNet121

This repository contains the complete implementation for Fairness & Explainability Analysis of DenseNet121, focusing on fairness evaluation, bias mitigation, and explainability methods for a DenseNet121 model trained on the ChestX-ray14 dataset.

## Repository Structure

```
.
├── src/                      # Core implementation
│   ├── config.py            # Configuration dataclass
│   ├── data.py              # Dataset and data loading
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation utilities (fairness metrics, group analysis)
│   ├── calibrate.py         # Calibration metrics (ECE, Brier, Temperature Scaling)
│   ├── uncertainty.py       # Uncertainty estimation (MC-Dropout, TTA)
│   ├── metrics.py           # Performance metrics (AUROC, AUPRC, fairness gaps)
│   ├── plots.py             # Visualization utilities (ROC curves, calibration plots)
│   ├── corruptions.py       # Image corruption functions
│   └── utils.py             # Helper functions (Grad-CAM, IG, case selection)
├── notebooks/               # Jupyter notebooks
│   ├── baseline.ipynb      # Initial model evaluation
│   ├── train.ipynb         # Training notebook
│   └── evaluate.ipynb      # **MAIN: All Tasks A-E implemented here**
│       └── outputs/        # Saved predictions and visualizations
├── report/                  # Assignment documentation
│   └── assignment_report.md # Comprehensive report covering all tasks
├── models/                  # Saved model checkpoints
│   └── densenet121.pth     # Pre-trained DenseNet121 weights
├── data/                    # Dataset directory (ChestX-ray14)
│   └── Chest14/
│       ├── Data_Entry_2017.csv
│       ├── images_001/ through images_012/
│       └── train_val_list.txt, test_list.txt
└── requirements.txt         # Python dependencies
```

## What You Need

1. **ChestX-ray14 Dataset**:
   - 112,120 frontal-view chest X-ray images (PNG format)
   - Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC
   - Metadata CSV: `Data_Entry_2017.csv`
   - 14 disease labels: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

2. **Python Environment**:
   - Python 3.13+
   - PyTorch 2.0+ with CUDA/MPS support
   - See `requirements.txt` for complete dependencies

3. **Hardware**:
   - GPU recommended (CUDA or Apple MPS)
   - Minimum 16GB RAM
   - ~50GB disk space for dataset + models

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/hasaniqbal777/Failure-Analysis-of-Medical-AI-Systems.git
cd Failure-Analysis-of-Medical-AI-Systems
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download ChestX-ray14 Dataset

```bash
# Download from NIH website and extract to data/Chest14/
# Expected structure:
# data/Chest14/
#   ├── Data_Entry_2017.csv
#   ├── images_001/images/
#   ├── images_002/images/
#   ├── ...
#   └── images_012/images/
```

## Training

### Option 1: Using Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/train.ipynb
```

Edit the configuration cell and run all cells sequentially.

### Option 2: Using Python Script

```bash
python src/train.py
```

**Training Configuration** (from `src/train.py`):
- **Model**: DenseNet121 (pretrained on ImageNet)
- **Batch size**: 16
- **Learning rate**: 0.001
- **Optimizer**: Adam (weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 20 (with early stopping, patience=5)
- **Loss**: Binary Cross-Entropy with Logits
- **Image size**: 224×224
- **Seed**: 42
- **Training time**: ~45-60 minutes per epoch on M1 Pro / RTX 3090

## Running the Assignment

### Main Evaluation Notebook (All Tasks A-E)

```bash
jupyter notebook notebooks/evaluate.ipynb
```

The evaluation notebook implements all five assignment tasks:

### **Task A: Data Audit & Bias Mapping (15%)**
- Gender distribution analysis (978F / 1,310M)
- Pneumonia prevalence by group (0.46% F vs 1.15% M)
- Bootstrap confidence intervals (1000 iterations)
- Three bias hypotheses:
  1. **Representation Bias**: Gender imbalance in dataset
  2. **Confounding Bias**: Age-disease correlation
  3. **Measurement Bias**: Acquisition protocol variation

### **Task B: Fairness Evaluation (30%)**
- **Metrics**: Demographic Parity (DP), Equal Opportunity (EO), Equalized Odds (EOds)
- **Baseline Results**: 28.57pp EO gap between male and female patients
- **Analysis**: ROC curves by group, calibration assessment, threshold sensitivity
- **Visualization**: Per-group performance tables, fairness gap plots

### **Task C: Mitigation Strategies (30%)**
- **Method 1**: Group-specific thresholds (female: 0.0273, male: 0.0584)
  - Reduces EO gap to ~0pp
  - Computational cost: <1 second
  - Trade-off: Slight macro-averaged performance reduction
- **Method 2**: Sample reweighting (simulated analysis)
- **Comparison**: Runtime, fairness improvement, performance impact
- **Visualizations**: 4-panel comparative analysis, regression plots, cost-benefit scoring

### **Task D: Explainability Methods (20%)**
- **Methods**: Grad-CAM (gradient-based) vs Integrated Gradients (path-based)
- **Sanity Tests**: 
  - Model randomization test (post-training layer ablation)
  - Data randomization test (label permutation sensitivity)
- **Stability Analysis**: Perturbation robustness (Gaussian noise, ±5% brightness)
- **Localization**: Bounding box overlap assessment (where available)
- **Comparison Visualization**: Side-by-side heatmaps with statistical analysis

### **Task E: Fairness-Explainability Interplay (5%)**
- **Clinical Case Studies**: 4 cases (TP/FN × Female/Male)
- **Attribution Analysis**: Gender-specific patterns in 40 pneumonia cases
- **Statistical Tests**: t-tests for concentration and entropy differences
- **Key Finding**: Post-mitigation explanation shifts may reveal learned biases
- **Visualizations**: 4×4 case study grid, attribution pattern comparisons

## Configuration

Edit `src/config.py` or the notebook configuration cells:

```python
cfg = Config(
    data_dir="/path/to/data/Chest14",
    csv_path="/path/to/data/Chest14/Data_Entry_2017.csv",
    checkpoint_path="models/densenet121.pth",
    target_classes=CHEST_XRAY14_CLASSES,  # All 14 classes
    primary_label="Pneumonia",            # For detailed analysis
    img_size=224,
    batch_size=32,
    num_workers=8,
    seed=1337
)
```

## Key Results

### Fairness Metrics (Task B)
- **Baseline Equal Opportunity Gap**: 28.57 percentage points (pp)
  - Female TPR: 42.86% (3/7 pneumonia cases detected)
  - Male TPR: 71.43% (10/14 pneumonia cases detected)
- **Demographic Parity Gap**: 12.50pp
- **Equalized Odds Gap**: 28.57pp

### Mitigation Effectiveness (Task C)
- **Method 1 (Group-Specific Thresholds)**:
  - Female threshold: 0.0273
  - Male threshold: 0.0584
  - Post-mitigation EO gap: ~0pp (complete closure)
  - Trade-off: Slight increase in overall FPR, minimal AUROC impact
- **Method 2 (Sample Reweighting)**: Conceptual analysis comparing fairness-performance trade-offs

### Explainability Comparison (Task D)
- **Grad-CAM**: Layer-level sensitivity, faster computation, coarser localization
- **Integrated Gradients**: Pixel-level attribution, path-dependent, finer detail
- **Sanity Test Results**:
  - Model randomization: Both methods show >60% attribution decrease (pass)
  - Data randomization: >50% attribution change (pass)
- **Stability**: Both methods robust to ±5% brightness perturbations

### Clinical Impact (Task E)
- Gender-based explanation differences detected in attribution concentration and entropy
- Post-mitigation, explanation patterns shift, suggesting learned gender biases
- Clinical recommendation: Audit model decisions for female patients with high-confidence predictions

## Reproducibility

### Hardware & Environment
- **Device**: Apple M1/M2 (MPS) or NVIDIA GPU (CUDA)
- **Python**: 3.10+
- **PyTorch**: 2.0+ with MPS/CUDA backend

### Model Details
- **Architecture**: DenseNet121 (ImageNet pretrained)
- **Dataset**: ChestX-ray14 (2,288 test images, 978F / 1,310M)
- **Primary Task**: Pneumonia detection (binary classification)
- **Image size**: 224×224 pixels
- **Normalization**: ImageNet statistics

### Evaluation Configuration
- **Fairness Metrics**: DP, EO, EOds with bootstrapped confidence intervals
- **Explainability**: Grad-CAM (features.denseblock4) + Integrated Gradients
- **Statistical Tests**: Two-sample t-tests, Welch's t-test for unequal variance
- **Visualization**: Matplotlib-based heatmaps, overlays, and comparative plots

### Random Seeds
All experiments use fixed seeds for reproducibility:
- **Global seed**: 1337
- PyTorch: `torch.manual_seed(1337)`
- NumPy: `np.random.seed(1337)`

### Expected Results
- **Baseline AUROC**: 0.887 (pneumonia-specific)
- **Baseline EO gap**: ~28pp
- **Mitigated EO gap**: <1pp (Method 1)
- **Explanation sanity tests**: Pass (>50% change on randomization)

## Key Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
Pillow>=10.0.0
jupyter>=1.0.0
```

See `requirements.txt` for the complete list.

## References

### Key Papers
1. **ChestX-ray14 Dataset**: Wang et al., "ChestX-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases", CVPR 2017
2. **Grad-CAM**: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization", ICCV 2017
3. **Integrated Gradients**: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017
4. **Fairness in Medical AI**: Chen et al., "Ethical Machine Learning in Healthcare", Annual Review of Biomedical Data Science 2021
5. **Sanity Checks**: Adebayo et al., "Sanity Checks for Saliency Maps", NeurIPS 2018

## License

MIT License - see LICENSE file for details.

## Contact

**Student**: Hasan Iqbal  
**Email**: hasaniqbal777@gmail.com  
**GitHub**: [@hasaniqbal777](https://github.com/hasaniqbal777)

For questions about this assignment implementation, please open an issue or contact via email.
