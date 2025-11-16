#!/usr/bin/env python3
"""Generate visualizations from training/benchmark results.

Saves figures to results/figures/:
 - training_curves.png
 - metrics_comparison.png
 - network_sample.png (if networkx installed)

Run with: ./cpdb/bin/python3 scripts/visualize_results.py
"""
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    sns = None

try:
    import networkx as nx
except Exception:
    nx = None


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(RESULTS, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def plot_training_curves(history_csv: str):
    df = pd.read_csv(history_csv)
    if 'epoch' in df.columns:
        df['epoch'] = df['epoch'].astype(int)

    models = df['model'].unique()

    plt.figure(figsize=(10, 4))
    for model in models:
        mdf = df[df['model'] == model].sort_values('epoch')
        if 'loss' in mdf.columns:
            plt.plot(mdf['epoch'], mdf['loss'], label=f"{model} loss", marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()
    try:
        plt.xlim(1, 14)
    except Exception:
        pass
    out1 = os.path.join(FIGDIR, 'training_loss.png')
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    for model in models:
        mdf = df[df['model'] == model].sort_values('epoch')
        if 'roc_auc' in mdf.columns:
            plt.plot(mdf['epoch'], mdf['roc_auc'], label=f"{model} ROC-AUC", marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('ROC-AUC')
    try:
        all_roc = df['roc_auc'].dropna().values
        rmin, rmax = float(all_roc.min()), float(all_roc.max())
        rpad = max(0.01, (rmax - rmin) * 0.15)
        plt.ylim(max(0.0, rmin - rpad), min(1.0, rmax + rpad))
    except Exception:
        plt.ylim(0.94, 1.0)
    try:
        plt.xlim(1, 14)
    except Exception:
        pass
    plt.title('Validation ROC-AUC per Epoch')
    plt.legend()
    out2 = os.path.join(FIGDIR, 'training_roc_auc.png')
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()

    return [out1, out2]


def plot_metrics_comparison(metrics_csv: str, benchmarks_csv: str = None):
    metrics = pd.read_csv(metrics_csv)
    metrics = metrics.set_index('model')

    m = metrics.reset_index().melt(id_vars='model', var_name='metric', value_name='value')

    plt.figure(figsize=(10, 5))
    if sns is not None:
        sns.barplot(data=m, x='metric', y='value', hue='model', palette='Set2')
    else:
        metrics_list = sorted(m['metric'].unique().tolist())
        models = sorted(m['model'].unique().tolist())
        n_metrics = len(metrics_list)
        x = range(n_metrics)
        width = 0.35
        for i, model in enumerate(models):
            vals = [m[(m['metric'] == mm) & (m['model'] == model)]['value'].values[0] if len(m[(m['metric'] == mm) & (m['model'] == model)]) > 0 else 0 for mm in metrics_list]
            plt.bar([xi + i * width for xi in x], vals, width=width, label=model)
        plt.xticks([xi + width / 2 for xi in x], metrics_list, rotation=45)

    try:
        vals = m['value'].astype(float).values
        vmin, vmax = float(vals.min()), float(vals.max())
        pad = max(0.01, (vmax - vmin) * 0.15)
        plt.ylim(max(0.0, vmin - pad), min(1.02, vmax + pad))
    except Exception:
        plt.ylim(0.94, 1.02)
    plt.ylabel('Score')
    plt.title('Final Metrics Comparison')
    plt.legend()
    plt.tight_layout()
    out = os.path.join(FIGDIR, 'metrics_comparison.png')
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def plot_network_sample(edge_csv: str, top_n: int = 30):
    if nx is None:
        return None

    df = pd.read_csv(edge_csv)
    if 'combined_weight_raw' in df.columns:
        weight_col = 'combined_weight_raw'
    elif 'weight' in df.columns:
        weight_col = 'weight'
    else:
        weight_col = None

    agg = df.groupby(['source_cell_type', 'target_cell_type'])[weight_col].sum().reset_index()
    agg_sorted = agg.sort_values(weight_col, ascending=False).head(top_n)

    G = nx.DiGraph()
    for _, row in agg_sorted.iterrows():
        s = row['source_cell_type']
        t = row['target_cell_type']
        w = float(row[weight_col])
        G.add_node(s, bipartite=0)
        G.add_node(t, bipartite=1)
        G.add_edge(s, t, weight=w)

    plt.figure(figsize=(12, 9))
    try:
        pos = nx.spring_layout(G, k=0.5, seed=42)
    except Exception:
        pos = nx.kamada_kawai_layout(G)

    degrees = dict(G.degree(weight='weight'))
    maxdeg = max(degrees.values()) if degrees else 1
    node_sizes = [300 + (degrees[n] / maxdeg) * 1200 for n in G.nodes()]

    weights = [d['weight'] for _, _, d in G.edges(data=True)]
    if weights:
        maxw = max(weights)
    else:
        maxw = 1.0
    edge_widths = [0.5 + (w / maxw) * 4.0 for w in weights]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='tab:blue', alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowstyle='->', arrowsize=12, edge_color='gray')

    plt.title(f'Top {top_n} cell-type → cell-type aggregated interaction weights')
    plt.axis('off')
    out = os.path.join(FIGDIR, 'network_sample_celltype_pairs.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    return out


def main():
    history_csv = os.path.join(RESULTS, 'training_history.csv')
    metrics_csv = os.path.join(RESULTS, 'training_metrics.csv')
    benchmarks_csv = os.path.join(RESULTS, 'benchmarks.csv')
    edge_csv = os.path.join(RESULTS, 'edge_list.csv')

    outs = []
    if os.path.exists(history_csv):
        outs += plot_training_curves(history_csv)

    if os.path.exists(metrics_csv):
        outs.append(plot_metrics_comparison(metrics_csv, benchmarks_csv if os.path.exists(benchmarks_csv) else None))

    if os.path.exists(edge_csv):
        net_out = plot_network_sample(edge_csv, top_n=30)
        if net_out:
            outs.append(net_out)

    for o in outs:
        if o:
            print(o)


if __name__ == '__main__':
    main()
