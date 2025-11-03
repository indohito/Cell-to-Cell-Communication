import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import os

adata = sc.read_h5ad('results/combined_dataset.h5ad')
counts = adata.obs['cell_type'].value_counts().sort_values(ascending=True)
os.makedirs('results/figures', exist_ok=True)

# Save file (optional)
counts.to_csv('results/cell_type_counts.csv', header=['count'])

# Draw a bar chart (horizontal)
plt.figure(figsize=(8, max(4, len(counts) * 0.25)))
counts.plot(kind='barh', color='C0')
plt.xlabel('Cell count')
plt.title('Cell counts per cell_type')
plt.tight_layout()
plt.savefig('results/figures/cell_type_counts.png', dpi=150)
print('Wrote results/cell_type_counts.csv and results/figures/cell_type_counts.png')

mapping = {
    # Immune
    'T.cells.CD4': 'Immune', 'T.cells.CD8': 'Immune', 'NK.cells': 'Immune',
    'NKT.cells': 'Immune', 'B.cells.Memory': 'Immune', 'B.cells.Naive': 'Immune',
    'Plasmablasts': 'Immune', 'Monocyte': 'Immune', 'Macrophage': 'Immune', 'DCs': 'Immune',
    'Cycling.T.cells': 'Immune',
    # Stromal
    'CAFs.myCAF.like': 'Stromal', 'CAFs.MSC.iCAF.like': 'Stromal', 'Myoepithelial': 'Stromal',
    # Endothelial
    'Endothelial.ACKR1': 'Endothelial', 'Endothelial.RGS5': 'Endothelial',
    'Endothelial.CXCL12': 'Endothelial', 'Endothelial.Lymphatic.LYVE1': 'Endothelial',
    # Tumor / epithelial
    'Cancer.Cycling': 'Tumor', 'Cancer.Basal.SC': 'Tumor', 'Cancer.Her2.SC': 'Tumor',
    'Cancer.LumA.SC': 'Tumor', 'Cancer.LumB.SC': 'Tumor',
    'Luminal.Progenitors': 'Epithelial', 'Mature.Luminal': 'Epithelial',
    # PVL / cycling / other
    'PVL.Differentiated': 'PVL', 'PVL.Immature': 'PVL',
    'Cycling.Myeloid': 'Cycling', 'Cycling.PVL': 'Cycling',
    # fallback: keep original if not mapped
}

# apply mapping with fallback to original
adata.obs['macro_type'] = adata.obs['cell_type'].map(mapping).fillna(adata.obs['cell_type'])

# produce counts per macro_type
counts_macro = adata.obs['macro_type'].value_counts().sort_values(ascending=True)
os.makedirs('results/figures', exist_ok=True)
counts_macro.to_csv('results/cell_type_macro_counts.csv', header=['count'])

# horizontal barplot
plt.figure(figsize=(8, max(3, len(counts_macro)*0.4)))
counts_macro.plot(kind='barh', color='C1')
plt.xlabel('Cell count')
plt.title('Counts per macro cell class')
plt.tight_layout()
plt.savefig('results/figures/cell_type_macro_counts.png', dpi=150)
