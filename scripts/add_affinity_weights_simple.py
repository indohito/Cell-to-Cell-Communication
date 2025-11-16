#!/usr/bin/env python3
"""
Simple version: Add uniform weights to integration table (quick test).
Can be extended with BindingDB later.
"""
import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
INTEGRATION = os.path.join(ROOT, 'results', 'integration_table.csv')
OUTDIR = os.path.join(ROOT, 'results')
os.makedirs(OUTDIR, exist_ok=True)

def main():
    if not os.path.exists(INTEGRATION):
        return
    
    df = pd.read_csv(INTEGRATION)
    
    df['weight'] = 1.0
    df['KD_nM'] = np.nan
    
    outw = os.path.join(OUTDIR, 'integration_table_weighted.csv')
    df.to_csv(outw, index=False)
    
    edges = df[['cell_type_A', 'ligand_gene', 'cell_type_B', 'receptor_gene', 'expr_A_mean', 'expr_B_mean', 'weight']].copy()
    edges = edges.rename(columns={'cell_type_A': 'source_cell_type', 'cell_type_B': 'target_cell_type', 
                                   'ligand_gene': 'ligand', 'receptor_gene': 'receptor'})
    edge_path = os.path.join(OUTDIR, 'edge_list.csv')
    edges.to_csv(edge_path, index=False)
    
    agg = edges.groupby(['source_cell_type', 'target_cell_type']).agg(
        total_weight=('weight', 'sum'),
        mean_weight=('weight', 'mean'),
        n_edges=('weight', 'count')
    ).reset_index()
    agg_path = os.path.join(OUTDIR, 'celltype_pair_weights.csv')
    agg.to_csv(agg_path, index=False)

if __name__ == '__main__':
    main()
