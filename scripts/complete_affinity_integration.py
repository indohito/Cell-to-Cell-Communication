#!/usr/bin/env python3
"""
Complete pipeline to integrate BindingDB affinity data with CellPhoneDB interactions.
This script:
1. Maps gene symbols to UniProt IDs
2. Matches proteins with BindingDB affinity data
3. Creates affinity-weighted edge list for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Setup paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "cellphonedb_data" / "sources"

def load_uniprot_mapping():
    """Load gene symbol to UniProt ID mapping from CellPhoneDB data."""
    print("Loading UniProt mapping...")
    uniprot_file = DATA_DIR / "uniprot_synonyms.tsv"
    
    df = pd.read_csv(uniprot_file, sep='\t')
    
    # Create mapping: gene name -> UniProt ID
    # Use both primary and synonym gene names
    mapping = {}
    
    for _, row in df.iterrows():
        uniprot_id = row['Entry']
        
        # Add primary gene name
        if pd.notna(row['Gene Names (primary)']):
            primary = str(row['Gene Names (primary)']).split()[0]
            mapping[primary] = uniprot_id
        
        # Add synonym gene names
        if pd.notna(row['Gene Names (synonym)']):
            synonyms = str(row['Gene Names (synonym)']).split()
            for syn in synonyms:
                mapping[syn] = uniprot_id
    
    print(f"  Loaded {len(mapping)} gene symbol -> UniProt mappings")
    return mapping

def load_bindingdb_affinity():
    """Load BindingDB affinity data."""
    print("Loading BindingDB affinity data...")
    affinity_file = RESULTS_DIR / "bindingdb_affinity_by_uniprot.csv"
    
    df = pd.read_csv(affinity_file)
    
    # Create a simple affinity lookup: uniprot -> best_affinity_nM
    affinity_dict = dict(zip(df['uniprot'], df['best_affinity_nM']))
    
    print(f"  Loaded {len(affinity_dict)} UniProt IDs with affinity data")
    return affinity_dict

def load_current_edge_list():
    """Load current edge list."""
    print("Loading current edge list...")
    edge_file = RESULTS_DIR / "edge_list.csv"
    
    df = pd.read_csv(edge_file)
    print(f"  Loaded {len(df)} edges")
    print(f"  Columns: {list(df.columns)}")
    
    return df

def add_affinity_weights(edge_df, uniprot_mapping, affinity_dict):
    """
    Add affinity weights to edges by:
    1. Mapping ligand and receptor gene symbols to UniProt IDs
    2. Looking up affinity values in BindingDB
    3. Computing final weights (normalized affinities)
    """
    print("\nComputing affinity-weighted edges...")
    
    # Copy dataframe
    df = edge_df.copy()
    
    # Initialize affinity and weight columns
    df['ligand_uniprot'] = None
    df['receptor_uniprot'] = None
    df['affinity_nM'] = np.nan
    df['weight'] = 1.0  # Default weight
    
    stats = {
        'total_edges': len(df),
        'ligand_mapped': 0,
        'receptor_mapped': 0,
        'both_mapped': 0,
        'affinity_found': 0,
        'weight_updated': 0
    }
    
    # Process each edge
    for idx, row in df.iterrows():
        ligand_gene = row['ligand']
        receptor_gene = row['receptor']
        
        # Map to UniProt
        ligand_up = uniprot_mapping.get(ligand_gene)
        receptor_up = uniprot_mapping.get(receptor_gene)
        
        if ligand_up:
            stats['ligand_mapped'] += 1
            df.at[idx, 'ligand_uniprot'] = ligand_up
        
        if receptor_up:
            stats['receptor_mapped'] += 1
            df.at[idx, 'receptor_uniprot'] = receptor_up
        
        # If both mapped, try to find affinity
        if ligand_up and receptor_up:
            stats['both_mapped'] += 1
            
            # Look up affinity for ligand-receptor pair
            # Try both directions since binding could be either way
            affinity = None
            
            if ligand_up in affinity_dict:
                affinity = affinity_dict[ligand_up]
            elif receptor_up in affinity_dict:
                affinity = affinity_dict[receptor_up]
            
            if affinity is not None:
                stats['affinity_found'] += 1
                df.at[idx, 'affinity_nM'] = affinity
                
                # Convert affinity to weight
                # Lower Kd = stronger binding = higher weight
                # Use: weight = 1 / (1 + affinity_nM / 100)
                # This maps: very low Kd (e.g. 0.1 nM) -> ~1.0 weight
                #            moderate Kd (100 nM) -> ~0.5 weight
                #            high Kd (10000 nM) -> ~0.01 weight
                weight = 1.0 / (1.0 + affinity / 100.0)
                df.at[idx, 'weight'] = weight
                stats['weight_updated'] += 1
        
        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1}/{len(df)} edges...")
    
    print(f"\nAffinity matching statistics:")
    print(f"  Total edges: {stats['total_edges']}")
    print(f"  Ligand genes mapped: {stats['ligand_mapped']} ({100*stats['ligand_mapped']/stats['total_edges']:.2f}%)")
    print(f"  Receptor genes mapped: {stats['receptor_mapped']} ({100*stats['receptor_mapped']/stats['total_edges']:.2f}%)")
    print(f"  Both mapped: {stats['both_mapped']} ({100*stats['both_mapped']/stats['total_edges']:.2f}%)")
    print(f"  Affinity found: {stats['affinity_found']} ({100*stats['affinity_found']/stats['total_edges']:.2f}%)")
    print(f"  Weights updated: {stats['weight_updated']}")
    
    return df, stats

def main():
    print("=" * 70)
    print("Complete Affinity Integration Pipeline")
    print("=" * 70)
    
    # Load data
    uniprot_mapping = load_uniprot_mapping()
    affinity_dict = load_bindingdb_affinity()
    edge_df = load_current_edge_list()
    
    # Add affinity weights
    edge_df_weighted, stats = add_affinity_weights(edge_df, uniprot_mapping, affinity_dict)
    
    # Save updated edge list
    output_file = RESULTS_DIR / "edge_list_with_affinity.csv"
    edge_df_weighted.to_csv(output_file, index=False)
    print(f"\nSaved affinity-weighted edge list to: {output_file}")
    
    # Print summary
    print(f"\nWeight distribution:")
    print(edge_df_weighted['weight'].describe())
    
    print(f"\nTop 10 highest-weight edges:")
    print(edge_df_weighted.nlargest(10, 'weight')[['source_cell_type', 'target_cell_type', 
                                                     'ligand', 'receptor', 'affinity_nM', 'weight']])
    
    print(f"\nSample edges with affinity data:")
    with_affinity = edge_df_weighted[edge_df_weighted['affinity_nM'].notna()].head(10)
    print(with_affinity[['ligand', 'receptor', 'affinity_nM', 'weight']])
    
    print("\n" + "=" * 70)
    print("Affinity integration complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
