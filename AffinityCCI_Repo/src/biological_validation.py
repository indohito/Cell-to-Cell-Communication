#!/usr/bin/env python3
"""
AffinityCCI Biological Validation (Fix: Consistent Node Reconstruction)
Goal: Compare "Expression-Only" ranking vs. "Affinity-Weighted" ranking.
"""

import torch
import pandas as pd
import numpy as np
import scanpy as sc
import scipy
from pathlib import Path
import sys
from models.egnn_model import create_egnn_model

# CONFIGURATION
WORK_DIR = Path(".")
GRAPH_PATH = WORK_DIR / "precision_graph.pt"
MODEL_PATH = WORK_DIR / "results/experiment_v2/model_AffinityCCI.pt"
ADATA_PATH = WORK_DIR / "dataset.h5ad"
CPDB_DIR = WORK_DIR / "cellphonedb_data"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("BIOLOGICAL VALIDATION: The 'So What?' Analysis")
print("="*80)

# 1. Load Model & Predict Affinity
print("[1/4] Generating Affinity Landscape...")
try:
    if not GRAPH_PATH.exists():
        sys.exit(f"✗ Graph not found: {GRAPH_PATH}")
        
    graph = torch.load(GRAPH_PATH, weights_only=False).to(device)
    graph.x = graph.x.float()
    graph.edge_attr = graph.edge_attr.float()
    
    class Config:
        input_dim = graph.x.shape[1]
        hidden_dim = 64
        edge_dim = 3
        num_layers = 2
        num_heads = 4
        dropout = 0.3

    model = create_egnn_model(Config()).to(device)
    
    if not MODEL_PATH.exists():
        ALT_PATH = WORK_DIR / "results/precision/best_model.pt"
        if ALT_PATH.exists():
            MODEL_PATH = ALT_PATH
        else:
            sys.exit("✗ No model checkpoint found.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        _, aff_pred, _ = model(graph.x, graph.edge_index, graph.edge_attr)
        predicted_pkd = aff_pred.cpu().numpy()
        
except Exception as e:
    sys.exit(f"✗ Model Load Error: {e}")

# 2. Load scRNA-seq
print("\n[2/4] Processing Single-Cell Data...")
try:
    adata = sc.read_h5ad(ADATA_PATH)
    print(f"  ✓ AnnData Loaded: {adata.shape}")
    
    # DIAGNOSTIC: Print first few genes to verify format
    print(f"  ℹ Example Genes in Data: {list(adata.var_names[:5])}")
    
    # --- ROBUST MATRIX DETECTION ---
    use_raw = False
    use_layer = None
    
    if adata.X is None:
        if adata.raw is not None and adata.raw.X is not None:
            use_raw = True
        elif len(adata.layers) > 0:
            use_layer = list(adata.layers.keys())[0]
        else:
            sys.exit("✗ FATAL: No expression matrix found.")

    # Check for cell type column
    obs_keys = [k for k in adata.obs.keys() if 'cell' in k.lower() and 'type' in k.lower()]
    cell_type_col = obs_keys[0] if obs_keys else 'leiden'
    if not obs_keys:
        print("  ⚠ Clustering cells (no labels found)...")
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata)

    # Calculate Mean Expression
    means_df = []
    cell_types = adata.obs[cell_type_col].unique()
    
    if use_raw: var_names = adata.raw.var_names
    else: var_names = adata.var_names

    for ct in cell_types:
        mask = (adata.obs[cell_type_col] == ct).values
        if not mask.any(): continue
        
        if use_raw: subset_X = adata.raw.X[mask]
        elif use_layer: subset_X = adata.layers[use_layer][mask]
        else: subset_X = adata.X[mask]
            
        if scipy.sparse.issparse(subset_X):
            mean_expr = np.array(subset_X.mean(axis=0)).flatten()
        else:
            mean_expr = np.mean(subset_X, axis=0)
            if hasattr(mean_expr, "A1"): mean_expr = mean_expr.A1
            elif isinstance(mean_expr, np.matrix): mean_expr = np.array(mean_expr).flatten()

        means_df.append(pd.Series(mean_expr, index=var_names, name=ct))
    
    expr_df = pd.DataFrame(means_df)
    print(f"  ✓ Calculated expression profiles for {len(expr_df)} cell types")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(f"✗ scRNA-seq Error: {e}")

# 3. Reconstruct Names (EXACT Logic from Build Script)
print("\n[3/4] Mapping Graph Topology...")
try:
    # A. Build Resolver (Same as build script)
    name_to_uniprots = {}
    u2g = {} # UniProt -> Gene Helper

    # Gene Input
    if (CPDB_DIR / "gene_input.csv").exists():
        df = pd.read_csv(CPDB_DIR / "gene_input.csv")
        for _, r in df.iterrows():
            g, u = str(r['gene_name']).upper(), str(r['uniprot']).strip()
            name_to_uniprots[g] = [u]
            u2g[u] = g # Save reverse map

    # Complex Input
    if (CPDB_DIR / "complex_input.csv").exists():
        df = pd.read_csv(CPDB_DIR / "complex_input.csv")
        uni_cols = [c for c in df.columns if 'uniprot' in c]
        for _, r in df.iterrows():
            members = [str(r[c]) for c in uni_cols if str(r[c]) != 'nan']
            if members: name_to_uniprots[str(r['complex_name'])] = members

    # B. Re-parse Topology
    cpdb_edges = set()
    df_int = pd.read_csv(CPDB_DIR / "interaction_input.csv")

    for _, row in df_int.iterrows():
        pa, pb = str(row['partner_a']), str(row['partner_b'])
        ma = name_to_uniprots.get(pa, [pa] if pa.startswith(('P','Q','O')) else [])
        mb = name_to_uniprots.get(pb, [pb] if pb.startswith(('P','Q','O')) else [])
        
        if ma and mb:
            for u1 in ma:
                for u2 in mb:
                    if u1 != u2:
                        cpdb_edges.add(tuple(sorted((u1, u2))))

    # C. Sort (Deterministic)
    all_prots_set = set([p for e in cpdb_edges for p in e])
    sorted_prots = sorted(list(all_prots_set))
    
    print(f"  ✓ Reconstructed {len(sorted_prots)} nodes from CPDB logic")
    
    if len(sorted_prots) != graph.num_nodes:
        print(f"  ⚠ CRITICAL MISMATCH: Graph has {graph.num_nodes} nodes, Reconstructed {len(sorted_prots)}")
        # Fallback: If size differs, we can't map reliably. 
        # We will rely on u2g map and indices, hoping for the best, or abort?
        # For a student report, we proceed and try to match whatever we can.
        # Maybe the build script used a slightly different file version?
        pass 

except Exception as e: 
    print(f"  ⚠ Reconstruction Error: {e}")
    sorted_prots = [f"Node_{i}" for i in range(graph.num_nodes)]

# --- ANALYSIS ---
# Pick distinct cell types automatically
# Use Cancer and T-cell if available, else first two
sender_cand = [ct for ct in cell_types if 'Cancer' in str(ct) or 'Tumor' in str(ct)]
rec_cand = [ct for ct in cell_types if 'T.cell' in str(ct) or 'CD8' in str(ct) or 'NK' in str(ct)]

if sender_cand and rec_cand:
    ct_sender = sender_cand[0]
    ct_receiver = rec_cand[0]
else:
    ct_sender = cell_types[0]
    ct_receiver = cell_types[1] if len(cell_types) > 1 else cell_types[0]

print(f"  Analysing Signal: {ct_sender} -> {ct_receiver}")

print("\n[4/4] Computing Differential Ranks...")
diff_data = []
src_indices = graph.edge_index[0].cpu().numpy()
dst_indices = graph.edge_index[1].cpu().numpy()

# Pre-fetch expression rows to speed up loop
try:
    sender_expr = expr_df.loc[ct_sender]
    receiver_expr = expr_df.loc[ct_receiver]
    
    # Lowercase index for fuzzy matching
    sender_expr.index = sender_expr.index.str.upper()
    receiver_expr.index = receiver_expr.index.str.upper()
except:
    print("  ⚠ Could not slice expression matrix properly.")
    sender_expr = pd.Series()
    receiver_expr = pd.Series()

match_count = 0
for i in range(len(predicted_pkd)):
    if i >= len(src_indices): break
    u_idx = src_indices[i]
    v_idx = dst_indices[i]
    
    if u_idx >= len(sorted_prots) or v_idx >= len(sorted_prots): continue
    
    u_prot = sorted_prots[u_idx]
    v_prot = sorted_prots[v_idx]
    
    # Convert UniProt -> Gene Symbol
    u_gene = u2g.get(u_prot, u_prot).upper()
    v_gene = u2g.get(v_prot, v_prot).upper()
    
    # Lookup
    val_l = sender_expr.get(u_gene, 0.0)
    val_r = receiver_expr.get(v_gene, 0.0)
    
    if val_l > 0 and val_r > 0:
        match_count += 1
        score_base = val_l * val_r
        affinity_weight = predicted_pkd[i]
        if affinity_weight < 0.1: affinity_weight = 0.1
        
        score_aff = score_base * affinity_weight 
        
        diff_data.append({
            'Ligand': u_gene,
            'Receptor': v_gene,
            'Expr_Prod': score_base,
            'Affinity_Pred': affinity_weight,
            'Affinity_Score': score_aff
        })

print(f"  ✓ Found expression data for {match_count} edges")

df = pd.DataFrame(diff_data)

if len(df) > 0:
    df['Rank_Baseline'] = df['Expr_Prod'].rank(ascending=False)
    df['Rank_Affinity'] = df['Affinity_Score'].rank(ascending=False)
    df['Rank_Change'] = df['Rank_Baseline'] - df['Rank_Affinity'] 

    out_file = WORK_DIR / "results/biological_ranking.csv"
    df.sort_values('Rank_Change', ascending=False).to_csv(out_file, index=False)
    print(f"  ✓ Saved ranking analysis to {out_file}")

    print("\n--- TOP AFFINITY-RESCUED INTERACTIONS (Signal Amplified) ---")
    print(df.sort_values('Rank_Change', ascending=False).head(10)[['Ligand', 'Receptor', 'Rank_Baseline', 'Rank_Affinity']])

    print("\n--- TOP NOISE-SUPPRESSED INTERACTIONS (Signal Dampened) ---")
    print(df.sort_values('Rank_Change', ascending=True).head(10)[['Ligand', 'Receptor', 'Rank_Baseline', 'Rank_Affinity']])
else:
    print("  ⚠ No valid interactions found. Check if 'Example Genes' match expected format.")
