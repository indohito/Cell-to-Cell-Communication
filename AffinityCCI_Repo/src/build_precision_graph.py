#!/usr/bin/env python3
"""
Build Precision Graph (Hybrid Strategy)
Topology: CellPhoneDB (High Relevance)
Affinity: BindingDB (Exact) + IntAct Confidence (Proxy)
"""
import pandas as pd
import numpy as np
import torch
import scipy.sparse
import scanpy as sc
from torch_geometric.data import Data
from pathlib import Path
import sys
import re

print("="*80 + "\nPHASE 1: PRECISION GRAPH (HYBRID INJECTION)\n" + "="*80)

WORK_DIR = Path(".")
CPDB_DIR = WORK_DIR / "cellphonedb_data"
INTACT_FILE = WORK_DIR / "intact-micluster.txt"
BINDINGDB_FILE = WORK_DIR / "BindingDB_All.tsv"
ADATA_FILE = WORK_DIR / "dataset.h5ad"
OUTPUT_FILE = WORK_DIR / "precision_graph.pt"

# ============================================================================
# 1. Build ID Resolver
# ============================================================================
print("[1/6] Building ID Resolver...")
name_to_uniprots = {}

# Gene Input
if (CPDB_DIR / "gene_input.csv").exists():
    df = pd.read_csv(CPDB_DIR / "gene_input.csv")
    for _, r in df.iterrows():
        name_to_uniprots[str(r['gene_name']).upper()] = [str(r['uniprot']).strip()]

# Complex Input
if (CPDB_DIR / "complex_input.csv").exists():
    df = pd.read_csv(CPDB_DIR / "complex_input.csv")
    uni_cols = [c for c in df.columns if 'uniprot' in c]
    for _, r in df.iterrows():
        members = [str(r[c]) for c in uni_cols if str(r[c]) != 'nan']
        if members: name_to_uniprots[str(r['complex_name'])] = members

print(f"  ID Resolver ready ({len(name_to_uniprots)} entries)")

# ============================================================================
# 2. CellPhoneDB Topology (The Backbone)
# ============================================================================
print("\n[2/6] Parsing CellPhoneDB Topology...")
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

print(f"  Graph Topology: {len(cpdb_edges)} edges (CellPhoneDB)")

# ============================================================================
# 3. IntAct Injection (Proxy Affinity)
# ============================================================================
print("\n[3/6] Loading IntAct Confidence Scores...")
intact_scores = {}

def extract_uni(field):
    m = re.search(r'uniprotkb:([A-Z0-9]+)', field)
    return m.group(1) if m else None

if INTACT_FILE.exists():
    with open(INTACT_FILE, 'r') as f:
        next(f)
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 15: continue
            
            # ID Extraction (Col 0/1 or 2/3)
            ua = extract_uni(p[0]) or extract_uni(p[2])
            ub = extract_uni(p[1]) or extract_uni(p[3])
            
            if ua and ub:
                try: score = float(p[14].split(':')[-1])
                except: score = 0.5
                
                edge = tuple(sorted((ua, ub)))
                # Store MAX score
                intact_scores[edge] = max(intact_scores.get(edge, 0.0), score)
    print(f"  Loaded {len(intact_scores)} IntAct scores")

# ============================================================================
# 4. BindingDB Injection (Real Affinity)
# ============================================================================
print("\n[4/6] Loading BindingDB Affinity...")
bdb_map = {}
# Regex for aggressive UniProt finding
uni_pat = re.compile(r'[OVPQ][0-9][A-Z0-9]{3}[0-9]')

try:
    reader = pd.read_csv(BINDINGDB_FILE, sep='\t', chunksize=20000, header=0, on_bad_lines='skip', low_memory=False)
    # Identify columns once
    chunk = next(reader) 
    cols = [c.lower() for c in chunk.columns]
    aff_idxs = [i for i, c in enumerate(cols) if 'kd (nm)' in c or 'ki (nm)' in c]
    
    # Restart
    reader = pd.read_csv(BINDINGDB_FILE, sep='\t', chunksize=20000, header=0, on_bad_lines='skip', low_memory=False)
    
    for chunk in reader:
        for c in aff_idxs: chunk.iloc[:, c] = pd.to_numeric(chunk.iloc[:, c], errors='coerce')
        
        for _, row in chunk.iterrows():
            vals = [row.iloc[c] for c in aff_idxs if pd.notnull(row.iloc[c])]
            if not vals: continue
            best = min(vals)
            if best <= 0: continue
            pkd = -np.log10(best * 1e-9)
            
            # Find proteins
            prots = sorted(list(set(uni_pat.findall(str(row.values)))))
            if len(prots) >= 2:
                for i in range(len(prots)):
                    for j in range(i+1, len(prots)):
                        edge = tuple(sorted((prots[i], prots[j])))
                        bdb_map[edge] = max(bdb_map.get(edge, 0.0), pkd)
except: pass
print(f"  Loaded {len(bdb_map)} BindingDB scores")

# ============================================================================
# 5. Merge & Build
# ============================================================================
print("\n[5/6] Merging Data Sources...")
all_prots = sorted(list(set([p for e in cpdb_edges for p in e])))
p2i = {p: i for i, p in enumerate(all_prots)}

src, dst = [], []
aff_vals = []
mask_vals = []

bdb_matches = 0
intact_matches = 0

for u, v in cpdb_edges:
    src.extend([p2i[u], p2i[v]])
    dst.extend([p2i[v], p2i[u]])
    
    edge = tuple(sorted((u, v)))
    
    # Logic: BindingDB > IntAct > None
    if edge in bdb_map:
        val = bdb_map[edge]
        mask = 1.0
        bdb_matches += 1
    elif edge in intact_scores:
        val = intact_scores[edge]
        mask = 1.0 # Treat IntAct score as "Measured" for training
        intact_matches += 1
    else:
        val = 0.0
        mask = 0.0
        
    aff_vals.extend([val, val])
    mask_vals.extend([mask, mask])

total = len(cpdb_edges)
print(f"  Edges: {total}")
print(f"  BindingDB Exact Hits: {bdb_matches} ({bdb_matches/total*100:.2f}%)")
print(f"  IntAct Proxy Hits:    {intact_matches} ({intact_matches/total*100:.2f}%)")
print(f"  Total Coverage:       {(bdb_matches+intact_matches)/total*100:.2f}%")

# ============================================================================
# 6. Features & Save
# ============================================================================
print("\n[6/6] Loading Features & Saving...")
feats = torch.randn((len(all_prots), 128))
try:
    adata = sc.read_h5ad(ADATA_FILE)
    if scipy.sparse.issparse(adata.X): m = np.array(adata.X.mean(axis=0)).flatten()
    else: m = np.mean(adata.X, axis=0)
    m = (m - m.mean()) / (m.std() + 1e-6)
    g_map = dict(zip(adata.var_names, m))
    
    # UniProt->Gene reverse map
    u2g = {}
    for g, us in name_to_uniprots.items():
        for u in us: u2g[u] = g
            
    lst = []
    for p in all_prots:
        g = u2g.get(p, p)
        lst.append([g_map.get(g, 0.0)])
    feats = torch.tensor(lst, dtype=torch.float32)
except: pass

edge_index = torch.tensor([src, dst], dtype=torch.long)
edge_attr = torch.stack([torch.ones(len(aff_vals)), torch.tensor(aff_vals), torch.tensor(mask_vals)], dim=1)
graph = Data(x=feats, edge_index=edge_index, edge_attr=edge_attr)
torch.save(graph, OUTPUT_FILE)
print(f"\nSAVED: {OUTPUT_FILE}")