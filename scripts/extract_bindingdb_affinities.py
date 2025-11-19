#!/usr/bin/env python3
"""
Stream BindingDB_All.tsv and extract the best (minimum) affinity per UniProt target.

Outputs:
 - results/bindingdb_affinity_by_uniprot.csv (uniprot, best_affinity_nM, affinity_type, example_line)
 - results/bindingdb_uniprot_set.txt

Run with venv python: ./cpdb/bin/python3 scripts/extract_bindingdb_affinities.py
"""
import os
import re
import math
ROOT = os.path.dirname(os.path.dirname(__file__))
BINDINGDB = os.path.join(ROOT, 'BindingDB_All.tsv')
OUTDIR = os.path.join(ROOT, 'results')
os.makedirs(OUTDIR, exist_ok=True)


def parse_float(val):
    if not val:
        return None
    val = val.strip()
    # remove commas
    val = val.replace(',', '')
    # sometimes contains inequality signs like '>10000'
    m = re.match(r'[<>~=]*\s*([0-9]*\.?[0-9]+)', val)
    if not m:
        return None
    try:
        return float(m.group(1))
    except:
        return None


def main():
    if not os.path.exists(BINDINGDB):
        print('BindingDB file not found at', BINDINGDB)
        return

    uni_cols = []
    affinity_cols = {}  # name -> index
    best = {}  # uniprot -> (value_nM, affinity_column_name, example_line_no)
    counts = {}

    with open(BINDINGDB, 'r', encoding='utf-8', errors='replace') as fh:
        header = fh.readline().rstrip('\n').split('\t')
        for i, h in enumerate(header):
            if 'UniProt (SwissProt) Primary ID' in h or ('UniProt (SwissProt) Primary ID of Target' in h) or ('UniProt (SwissProt) Primary ID of Target Chain' in h):
                uni_cols.append(i)
            if h.strip() in ('Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'EC50 (nM)'):
                affinity_cols[h.strip()] = i
        if not uni_cols:
            # fuzzy match
            for i,h in enumerate(header):
                if 'UniProt' in h and 'Primary ID' in h:
                    uni_cols.append(i)
        print(f'Found {len(uni_cols)} UniProt columns and {len(affinity_cols)} affinity cols')

        for line_no, line in enumerate(fh, start=2):
            cols = line.rstrip('\n').split('\t')
            # gather affinities in this row
            affinities = []
            for name, idx in affinity_cols.items():
                if idx < len(cols):
                    val = parse_float(cols[idx])
                    if val is not None and not math.isnan(val):
                        affinities.append((name, val))
            if not affinities:
                continue
            # pick best (minimum nM)
            best_affinity_name, best_val = min(affinities, key=lambda x: x[1])

            # extract UniProt IDs from uni_cols
            for ci in uni_cols:
                if ci < len(cols):
                    val = cols[ci].strip()
                    if not val:
                        continue
                    # multiple IDs may be separated by ';' or '|'
                    tokens = re.split(r'[;|,]', val)
                    for tok in tokens:
                        tok = tok.strip()
                        if not tok:
                            continue
                        # update best
                        prev = best.get(tok)
                        counts[tok] = counts.get(tok, 0) + 1
                        if prev is None or best_val < prev[0]:
                            best[tok] = (best_val, best_affinity_name, line_no)

            if line_no % 500000 == 0:
                print(f'Processed {line_no} lines; distinct UniProt with affinities so far: {len(best)}')

    # write outputs
    import csv
    with open(os.path.join(OUTDIR, 'bindingdb_affinity_by_uniprot.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['uniprot', 'best_affinity_nM', 'affinity_type', 'example_line', 'num_records'])
        for up, (val, name, line_no) in sorted(best.items(), key=lambda x: x[0]):
            w.writerow([up, val, name, line_no, counts.get(up, 0)])

    with open(os.path.join(OUTDIR, 'bindingdb_uniprot_set.txt'), 'w', encoding='utf-8') as fh:
        for up in sorted(best.keys()):
            fh.write(up + '\n')

    print('Wrote bindingdb_affinity_by_uniprot.csv and bindingdb_uniprot_set.txt with', len(best), 'entries')


if __name__ == '__main__':
    main()
