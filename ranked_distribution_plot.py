import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# CONFIGURATION
FILE_NAME = 'combined_dataset.h5ad'
CELL_TYPE_COLUMN = 'cell_type'

def main():
    print(f"--- Generating Strictly Ranked Distribution Plot ---")
    adata = sc.read_h5ad(FILE_NAME)
    
    # 1. Get Counts and FORCE Sort (Most to Least)
    # sort=True ensures the highest numbers come first
    counts = adata.obs[CELL_TYPE_COLUMN].value_counts(sort=True, ascending=False)
    
    print("Top 5 Most Frequent:")
    print(counts.head())
    
    # 2. Create a simple Table for plotting
    df_counts = pd.DataFrame({
        'Cell Type': counts.index,
        'Count': counts.values
    })
    
    # 3. Dynamic Figure Size
    # If you have 50 cell types, a standard plot is too short. 
    # This math makes the plot taller based on how many rows we have.
    num_types = len(counts)
    height = max(8, num_types * 0.3) 
    
    plt.figure(figsize=(12, height))
    
    # 4. Plot with explicit 'order'
    sns.barplot(
        data=df_counts,
        y='Cell Type',
        x='Count',
        palette='viridis',
        order=df_counts['Cell Type']  # <--- THIS LINE FIXES THE SORTING
    )
    
    plt.title(f"Node Distribution (Ranked): {num_types} Cell Types")
    plt.xlabel("Number of Cells")
    
    # Add labels to the end of the bars for clarity
    for i, v in enumerate(counts.values):
        plt.text(v + 5, i, str(v), color='black', va='center', fontsize=10)

    plt.tight_layout()
    
    save_name = "Graph_Node_Distribution_Ranked.png"
    plt.savefig(save_name)
    print(f"✅ Saved sorted plot: {save_name}")

if __name__ == "__main__":
    main()
