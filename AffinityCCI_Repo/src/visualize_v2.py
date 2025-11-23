import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config
WORK_DIR = Path(".")
CSV_FILE = WORK_DIR / "results" / "experiment_v2" / "metrics_v2.csv"
PLOT_DIR = WORK_DIR / "results" / "plots_v2"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def plot_experiment():
    if not CSV_FILE.exists():
        print("CSV file not found.")
        return

    df = pd.read_csv(CSV_FILE)
    sns.set_theme(style="whitegrid")
    
    metrics = [
        ("Affinity_MSE", "Affinity Prediction Error (MSE)", "Lower is Better"),
        ("ROC_AUC", "Classification ROC-AUC", "Higher is Better"),
        ("PR_AUC", "Precision-Recall AUC", "Higher is Better")
    ]
    
    for metric, title, note in metrics:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x="Epoch", y=metric, hue="Mode", linewidth=2.5, palette=["gray", "#d62728"])
        
        plt.title(f"{title}\n({note})", fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.xlabel("Epoch", fontsize=12)
        
        # Zoom in for MSE
        if metric == "Affinity_MSE":
            plt.ylim(0, 1.0)
            
        save_path = PLOT_DIR / f"{metric.lower()}_comparison.png"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved {save_path}")

if __name__ == "__main__":
    plot_experiment()
