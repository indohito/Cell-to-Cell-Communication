#!/usr/bin/env python3
"""
Compute affinity-based edge weights from integration_table.csv and produce edge lists and aggregates.

Outputs:
 - results/integration_table_weighted.csv
 - results/edge_list.csv
 - results/celltype_pair_weights.csv

Weight definitions:
 - weight_inv_nM = 1 / KD_nM
 - weight_pK = 9 - log10(KD_nM)  (approximate pKd when KD in nM)
 Both weights are normalized to 0-1 across the dataset (min-max) and stored as *_norm.

Run with venv python: ./venv/bin/python3 scripts/add_affinity_weights.py
"""
import os
import math
import pandas as pd
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
INTEGRATION = os.path.join(ROOT, 'results', 'integration_table.csv')
BINDINGDB = os.path.join(ROOT, 'BindingDB_All.tsv')
OUTDIR = os.path.join(ROOT, 'results')
os.makedirs(OUTDIR, exist_ok=True)


def safe_float(x):
    try:
        if x is None or pd.isna(x):
            return np.nan
        val = float(x)
        return val if val > 0 else np.nan
    except:
        return np.nan


def load_bindingdb_affinities():
    """Load BindingDB and build gene-pair -> affinity mapping."""
    if not os.path.exists(BINDINGDB):
        return {}
    
    affinity_map = defaultdict(list)
    processed = 0
    
    try:
        for chunk in pd.read_csv(BINDINGDB, sep='\t', low_memory=False, chunksize=50000):
            if 'Target Source Organism According to Curator or DataSource' in chunk.columns:
                chunk = chunk[chunk['Target Source Organism According to Curator or DataSource'].str.contains(
                    'Homo sapiens|Human', na=False, case=False, regex=True)]
            
            for _, row in chunk.iterrows():
                ligand_name = str(row.get('BindingDB Ligand Name', '')).strip()
                target_name = str(row.get('Target Name', '')).strip()
                
                if not ligand_name or not target_name:
                    continue
                
                kd = safe_float(row.get('Kd (nM)', np.nan))
                ki = safe_float(row.get('Ki (nM)', np.nan))
                ic50 = safe_float(row.get('IC50 (nM)', np.nan))
                
                affinities = [v for v in [kd, ki, ic50] if pd.notna(v)]
                if not affinities:
                    continue
                
                best_aff = min(affinities)
                affinity_map[(ligand_name, target_name)].append(best_aff)
                processed += 1
    
    except Exception as e:
        return {}
    
    affinity_best = {}
    for key, affinities in affinity_map.items():
        affinity_best[key] = min(affinities)
    
    return affinity_best


def main():
    if not os.path.exists(INTEGRATION):
        return
    
    df = pd.read_csv(INTEGRATION)
    affinity_map = load_bindingdb_affinities()
    
    kd_values = []
    matched_count = 0
    
    for idx, row in df.iterrows():
        ligand = str(row['ligand_gene']).strip()
        receptor = str(row['receptor_gene']).strip()
        
        kd = None
        
        if (ligand, receptor) in affinity_map:
            kd = affinity_map[(ligand, receptor)]
            matched_count += 1
        elif (receptor, ligand) in affinity_map:
            kd = affinity_map[(receptor, ligand)]
            matched_count += 1
        
        kd_values.append(kd)
    
    df['KD_nM'] = kd_values
    
    if df['KD_nM'].notna().sum() == 0:
        df['weight'] = 1.0
    else:
        df['weight_inv_nM'] = df['KD_nM'].apply(lambda x: 1.0 / x if pd.notna(x) and x > 0 else np.nan)
        
        def pK(x):
            if pd.isna(x) or x <= 0:
                return np.nan
            return 9.0 - math.log10(x)
        
        df['weight_pK'] = df['KD_nM'].apply(pK)
        
        for col in ['weight_inv_nM', 'weight_pK']:
            vals = df[col]
            mn = vals.min(skipna=True)
            mx = vals.max(skipna=True)
            if pd.isna(mn) or pd.isna(mx) or mx == mn:
                df[col + '_norm'] = 0.5
            else:
                df[col + '_norm'] = vals.apply(lambda v: (v - mn) / (mx - mn) if pd.notna(v) else 0.5)
        
        df['weight'] = df['weight_pK_norm'].where(df['weight_pK_norm'].notna() & (df['weight_pK_norm'] > 0), 0.5)
    
    outw = os.path.join(OUTDIR, 'integration_table_weighted.csv')
    df.to_csv(outw, index=False)
    
    edge_cols = ['cell_type_A', 'ligand_gene', 'cell_type_B', 'receptor_gene', 'weight']
    if 'KD_nM' in df.columns:
        edge_cols.append('KD_nM')
    
    edges = df[edge_cols].copy()
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
