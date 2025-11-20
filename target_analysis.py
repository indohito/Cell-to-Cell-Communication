import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CONFIGURATION
FILE_NAME = 'combined_dataset.h5ad'

def main():
    print(f"--- Loading {FILE_NAME} ---")
    adata = sc.read_h5ad(FILE_NAME)
    
    # 1. EXTRACT THE HIDDEN TABLE
    print("\n[1] Extracting 'ccc_target' Table...")
    df_targets = pd.DataFrame(adata.uns['ccc_target'])
    
    # Print a preview for the terminal
    print(f"   Total Pairs: {len(df_targets)}")
    print(df_targets.head(10))
    
    # Save it so you can see the gene names clearly later
    df_targets.to_csv("training_targets.csv", index=False)
    print("✅ Saved table to: training_targets.csv")

   # --- 2. ANALYZING THE RESPONSE (With Tags & Colors) ---
    print("\n[2] Analyzing the 'response' column...")
    
    # Create a new readable column for the plot
    df_targets['Label'] = df_targets['response'].map({0: 'Negative (0)', 1: 'Positive (1)'})
    
    plt.figure(figsize=(8, 6))
    
    # distinct colors: 0=Grey/Blue (Background), 1=Red (Signal)
    custom_palette = {'Negative (0)': '#a6cee3', 'Positive (1)': '#e31a1c'}
    
    ax = sns.countplot(data=df_targets, x='Label', palette=custom_palette, order=['Negative (0)', 'Positive (1)'])
    
    plt.title(f"Class Imbalance: {len(df_targets)} Total Pairs")
    plt.xlabel("Interaction Status")
    plt.ylabel("Count")

    # --- THE TAGGING PART (Add numbers on bars) ---
    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig("EDA_Target_Distribution_Tagged.png")
    print("✅ Saved plot: EDA_Target_Distribution_Tagged.png")

    # 3. TOP PLAYERS (Which Genes are most involved?)
    print("\n[3] Identifying Key Ligands and Receptors...")
    
    # Top 10 Ligands
    top_ligands = df_targets['ligand'].value_counts().head(10)
    # Top 10 Receptors
    top_targets = df_targets['target'].value_counts().head(10)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.barplot(y=top_ligands.index, x=top_ligands.values, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Most Frequent Ligands in Training Set")
    
    sns.barplot(y=top_targets.index, x=top_targets.values, ax=axes[1], palette="Reds_r")
    axes[1].set_title("Most Frequent Receptors (Targets)")
    
    plt.tight_layout()
    plt.savefig("EDA_Top_Genes.png")
    print("✅ Saved plot: EDA_Top_Genes.png")

if __name__ == "__main__":
    main()
