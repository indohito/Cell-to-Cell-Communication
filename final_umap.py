import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # <--- Added this missing line!

# CONFIGURATION
FILE_NAME = 'combined_dataset.h5ad'
CELL_TYPE_COLUMN = 'cell_type' 
N_TOP_CELL_TYPES = 10 

# MANUAL ADJUSTMENTS FOR LABEL PLACEMENT (X, Y offset from cluster center)
LABEL_OFFSETS = {
    'T.cells.CD4': (0.5, 0.5),       # Shift away from center
    'T.cells.CD8': (-0.5, 0.5),
    'Cancer.Cycling': (-0.5, -0.5),
    'CAFs.myCAF.like': (0.5, 0.5),
    'Cancer.Basal.SC': (0.5, -0.5),
    'Macrophage': (-0.5, 0.5),
    'B.cells.Memory': (0.5, 0.5),
    'Plasmablasts': (-0.5, 0.5),
    'Cancer.Her2.SC': (0.5, 0.5),
    'Monocyte': (-0.5, -0.5),
}

def main():
    print(f"--- Generating Final UMAP for Top {N_TOP_CELL_TYPES} Cell Types ---")
    adata = sc.read_h5ad(FILE_NAME)

    # 1. Identify the Top 10
    cell_counts = adata.obs[CELL_TYPE_COLUMN].value_counts()
    top_cell_types = cell_counts.head(N_TOP_CELL_TYPES).index.tolist()
    
    # Filter the data
    adata_filtered = adata[adata.obs[CELL_TYPE_COLUMN].isin(top_cell_types)].copy()
    
    # 2. Calculate UMAP if missing
    if 'X_umap' not in adata_filtered.obsm:
        print("   (Calculating UMAP for filtered subset...)")
        sc.pp.neighbors(adata_filtered, n_neighbors=15, n_pcs=30)
        sc.tl.umap(adata_filtered)
    else:
        print("   (Using existing UMAP coordinates)")

    # 3. PLOT: THE BEAUTIFUL UMAP
    print("   Generating Plot...")
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot points without legend
    sc.pl.umap(adata_filtered, 
               color=CELL_TYPE_COLUMN, 
               show=False, 
               ax=ax, 
               title=f"Top {N_TOP_CELL_TYPES} Cell Clusters (Final)", 
               frameon=False, 
               palette='tab20',
               legend_loc='none'
              )
    
    # Manually add labels
    umap_coords = pd.DataFrame(adata_filtered.obsm['X_umap'], index=adata_filtered.obs_names, columns=['umap_0', 'umap_1'])
    umap_coords[CELL_TYPE_COLUMN] = adata_filtered.obs[CELL_TYPE_COLUMN]
    
    # Calculate centroids
    cluster_centroids = umap_coords.groupby(CELL_TYPE_COLUMN)[['umap_0', 'umap_1']].mean()
    
    # Add text labels
    for cluster_name, centroid in cluster_centroids.iterrows():
        offset_x, offset_y = LABEL_OFFSETS.get(cluster_name, (0,0))
        
        ax.text(centroid['umap_0'] + offset_x, 
                centroid['umap_1'] + offset_y, 
                cluster_name, 
                fontsize=12, 
                fontweight='bold', 
                color='black', 
                ha='center', 
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", lw=0.5, alpha=0.6))
    
    plt.tight_layout()
    save_name = f"Final_UMAP_Top{N_TOP_CELL_TYPES}.png"
    plt.savefig(save_name, dpi=300)
    print(f"✅ Saved Final UMAP: {save_name}")
    
    # 4. PLOT: THE DISTRIBUTION
    plt.figure(figsize=(10, 6))
    sns.barplot(y=cell_counts.head(N_TOP_CELL_TYPES).index, 
                x=cell_counts.head(N_TOP_CELL_TYPES).values, 
                palette='viridis')
    plt.title(f"Counts of Top {N_TOP_CELL_TYPES} Cell Types")
    plt.xlabel("Number of Cells")
    plt.tight_layout()
    plt.savefig(f"Distribution_Top{N_TOP_CELL_TYPES}.png")
    print(f"✅ Saved Distribution: Distribution_Top{N_TOP_CELL_TYPES}.png")

if __name__ == "__main__":
    main()
