import numpy as np
import matplotlib.pyplot as plt

def plot_bar_with_deltas(names, baseline, variants, variant_names, title="Delta vs Clean (lower is worse)"):
    plt.figure()
    x = np.arange(len(names))
    width = 0.8 / (len(variants) + 1)
    plt.bar(x, baseline, width=width, label="Clean")
    for i, (var, vname) in enumerate(zip(variants, variant_names), start=1):
        plt.bar(x + i*width, var, width=width, label=vname)
    plt.xticks(x + width*len(variants)/2, names, rotation=30, ha='right')
    plt.title(title)
    plt.legend()
    plt.tight_layout()

def plot_risk_coverage(coverages, risks, title="Risk–Coverage"):
    """Plot risk-coverage curve for selective prediction.
    
    Shows the trade-off between coverage (% of predictions kept) and 
    error rate on the kept predictions. Ideal: low error maintained at high coverage.
    """
    plt.figure(figsize=(7,5))
    plt.plot(coverages, risks, marker='o', linewidth=2, markersize=6, label='Model')
    
    # Add reference line for random rejection (would maintain same error rate)
    if len(risks) > 0:
        baseline_error = risks[0]  # Error at 100% coverage
        plt.axhline(y=baseline_error, color='r', linestyle='--', alpha=0.5, label=f'No rejection (error={baseline_error:.3f})')
    
    plt.xlabel("Coverage (fraction of predictions kept)", fontsize=11)
    plt.ylabel("Error rate on kept predictions", fontsize=11)
    plt.title(title, fontsize=12)
    plt.xlim([0, 1.05])
    plt.ylim([0, max(risks)*1.1] if len(risks) > 0 else [0, 1])
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
