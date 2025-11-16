#!/usr/bin/env python3
"""
Build integration table merging:
1. TNBC single-cell expression data (cell type × genes)
2. CellPhoneDB ligand-receptor pairs

Output: results/integration_table.csv with columns:
  [cell_type_A, ligand_gene, cell_type_B, receptor_gene,
   expr_A_mean, expr_B_mean]

Note: BindingDB affinity will be added separately via add_affinity_weights.py
"""

import pandas as pd
import numpy as np
import scanpy as sc
import os
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, 'results')

def load_gene_to_uniprot_mapping():
    """Load gene name to UniProt ID mapping from CellPhoneDB."""
    mapping_path = os.path.join(ROOT, 'cellphonedb_data', 'gene_input.csv')
    df = pd.read_csv(mapping_path)
    
    uniprot_to_gene = dict(zip(df['uniprot'], df['hgnc_symbol']))
    gene_to_uniprot = dict(zip(df['hgnc_symbol'], df['uniprot'])) 
    return uniprot_to_gene, gene_to_uniprot

def load_tnbc_data():
    """Load TNBC dataset and compute mean expression per cell type."""
    dataset_path = os.path.join(ROOT, 'dataset.h5ad')
    adata = sc.read_h5ad(dataset_path)
    
    if adata.X is None:
        if 'normalized' in adata.layers:
            adata.X = adata.layers['normalized']
        elif 'counts' in adata.layers:
            adata.X = adata.layers['counts']
    
    cell_types = sorted(adata.obs['cell_type'].unique())
    expr_by_cell_type = {}
    
    for ct in cell_types:
        mask = adata.obs['cell_type'] == ct
        subset = adata[mask, :]
        
        if hasattr(subset.X, 'toarray'):
            mean_expr = subset.X.toarray().mean(axis=0)
        else:
            mean_expr = np.asarray(subset.X.mean(axis=0)).flatten()
        mean_expr = np.asarray(mean_expr).flatten()
        
        expr_by_cell_type[ct] = dict(zip(adata.var['feature_name'], mean_expr))
    
    return expr_by_cell_type, cell_types

def load_cellphonedb_pairs():
    """Load CellPhoneDB ligand-receptor pairs."""
    pairs_path = os.path.join(ROOT, 'cellphonedb_data', 'interaction_input.csv')
    df = pd.read_csv(pairs_path, low_memory=False)
    return df

def build_integration_table():
    """Build complete integration table."""
    uniprot_to_gene, gene_to_uniprot = load_gene_to_uniprot_mapping()
    expr_by_cell_type, cell_types = load_tnbc_data()
    cellphonedb_pairs = load_cellphonedb_pairs()
    
    rows = []
    total_pairs = 0
    complex_pairs = 0
    
    for idx, row in cellphonedb_pairs.iterrows():
        partner_a = str(row.get('partner_a', '')).strip()
        partner_b = str(row.get('partner_b', '')).strip()
        interactors = str(row.get('interactors', '')).strip()
        
        if partner_a in ['nan', ''] or partner_b in ['nan', '']:
            continue
        
        if 'complex' in partner_a.lower() and 'complex' in partner_b.lower():
            complex_pairs += 1
            continue
        
        genes_a = []
        genes_b = []
        
        if 'complex' in partner_a.lower() or 'complex' in partner_b.lower():
            if pd.notna(interactors) and interactors not in ['nan', '']:
                try:
                    parts = interactors.split('-')
                    if len(parts) == 2:
                        genes_a = parts[0].split('+')
                        genes_b = parts[1].split('+')
                except:
                    pass
        else:
            if partner_a in uniprot_to_gene:
                genes_a = [uniprot_to_gene[partner_a]]
            else:
                genes_a = [partner_a]
            
            if partner_b in uniprot_to_gene:
                genes_b = [uniprot_to_gene[partner_b]]
            else:
                genes_b = [partner_b]
        
        if not genes_a or not genes_b:
            continue
        
        for cell_type_a in cell_types:
            for cell_type_b in cell_types:
                for gene_a in genes_a:
                    for gene_b in genes_b:
                        gene_a_clean = gene_a.strip()
                        gene_b_clean = gene_b.strip()
                        
                        expr_a = expr_by_cell_type[cell_type_a].get(gene_a_clean, np.nan)
                        expr_b = expr_by_cell_type[cell_type_b].get(gene_b_clean, np.nan)
                        
                        if np.isnan(expr_a) or np.isnan(expr_b) or expr_a == 0 or expr_b == 0:
                            continue
                        
                        rows.append({
                            'cell_type_A': cell_type_a,
                            'ligand_gene': gene_a_clean,
                            'cell_type_B': cell_type_b,
                            'receptor_gene': gene_b_clean,
                            'expr_A_mean': float(expr_a),
                            'expr_B_mean': float(expr_b)
                        })
                        total_pairs += 1
    
    integration_table = pd.DataFrame(rows)
    
    if len(integration_table) > 0:
        output_path = os.path.join(RESULTS_DIR, 'integration_table.csv')
        integration_table.to_csv(output_path, index=False)
    
    return integration_table

if __name__ == '__main__':
    integration_table = build_integration_table()
