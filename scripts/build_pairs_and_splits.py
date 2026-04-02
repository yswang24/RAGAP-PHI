#!/usr/bin/env python3
"""
build_pairs_and_splits.py
把 raw pair 文件（VHDB 风格或简单 phage-host 列表）映射到 host_gcf 并做 train/val/test 划分。

Usage:
 python build_pairs_and_splits.py \
   --raw_pairs inputs/virus_host_with_GCF.tsv \
   --host_catalog artifacts/ragap_phi/catalogs/host_catalog.parquet \
   --taxonomy_parquet data/taxonomy/taxonomy_with_alias.parquet \
   --out_dir artifacts/ragap_phi/pairs \
   --split random --seed 13
"""
import argparse, os, csv, random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict

def parse_extract_gcfs(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, list):
        return [x.strip() for x in s if x]
    # sometimes semicolon-delimited string
    return [x.strip() for x in str(s).split(';') if x.strip()]

def choose_representative_gcf(gcf_list, host_catalog_set):
    # prefer one present in host_catalog; otherwise return first
    for g in gcf_list:
        if g in host_catalog_set:
            return g
    return gcf_list[0] if len(gcf_list) > 0 else None

def ensure_str(x):
    return str(int(x)) if (not pd.isna(x) and float(x).is_integer()) else str(x)

def build_pairs(raw_pairs_path, host_catalog_path, taxonomy_path, out_dir, src_colnames=None):
    os.makedirs(out_dir, exist_ok=True)
    host_df = pd.read_parquet(host_catalog_path)
    host_set = set(host_df['host_gcf'].astype(str).tolist())

    tax_df = pd.read_parquet(taxonomy_path)
    tax_set = set(tax_df['taxid'].astype(str).tolist())

    # load raw pairs (try tsv/csv)
    print("Reading raw pairs:", raw_pairs_path)
    rp = pd.read_csv(raw_pairs_path, sep=None, engine='python')
    rp_cols = rp.columns.tolist()
    print("Columns detected:", rp_cols)

    # try to adapt to common VHDB columns: refseq_id / host_taxid / Extracted_GCFs
    if 'refseq_id' in rp_cols:
        ph_col = 'refseq_id'
    elif 'phage_id' in rp_cols:
        ph_col = 'phage_id'
    elif 'virus_refseq' in rp_cols:
        ph_col = 'virus_refseq'
    else:
        raise RuntimeError("Cannot find phage id column in raw pairs. Provide file with 'refseq_id' or 'phage_id'.")

    if 'host_gcf' in rp_cols:
        host_gcf_col = 'host_gcf'
    elif 'Extracted_GCFs' in rp_cols:
        host_gcf_col = 'Extracted_GCFs'
    else:
        # maybe only host_taxid present
        host_gcf_col = None

    host_tax_col = None
    if 'host_taxid' in rp_cols:
        host_tax_col = 'host_taxid'
    elif 'host_tax_id' in rp_cols:
        host_tax_col = 'host_tax_id'

    records = []
    missing_gcf = []

    for _, r in rp.iterrows():
        phid = str(r[ph_col]).strip()
        species_taxid = str(r[host_tax_col]) if host_tax_col else None

        if host_gcf_col is not None and not pd.isna(r[host_gcf_col]):
            gcfs = parse_extract_gcfs(r[host_gcf_col])
        elif species_taxid is not None:
            # 尝试从 host_catalog 找到匹配
            matches = host_df[host_df['host_species_taxid'].astype(str) == species_taxid]
            gcfs = matches['host_gcf'].astype(str).tolist() if len(matches) > 0 else []
        else:
            gcfs = []

        if len(gcfs) == 0:
            missing_gcf.append((phid, species_taxid))
            continue

        # ⚡关键修改：每个 GCF 都生成一条记录
        for host_gcf in gcfs:
            if host_gcf not in host_set:
                continue
            records.append({
                'phage_id': phid,
                'host_gcf': host_gcf,
                'host_species_taxid': species_taxid,
                'label': 1,
                'source': os.path.basename(raw_pairs_path)
            })

    out_all = os.path.join(out_dir, "pairs_all.tsv")
    df = pd.DataFrame(records)
    df.to_csv(out_all, sep='\t', index=False)
    print("WROTE", out_all, "rows=", len(df))
    if len(missing_gcf) > 0:
        with open(os.path.join(out_dir, "missing_pairs_gcf.tsv"), "w") as f:
            for x,y in missing_gcf:
                f.write(f"{x}\t{y}\n")
        print("WROTE missing_pairs_gcf.tsv", len(missing_gcf))
    return df

def parse_ratios(raw):
    if raw is None:
        return (0.8, 0.1, 0.1)
    if isinstance(raw, (list, tuple)):
        values = [float(x) for x in raw]
    else:
        values = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if len(values) != 3:
        raise RuntimeError(f"Expected exactly 3 ratios, got: {raw}")
    total = sum(values)
    if total <= 0:
        raise RuntimeError(f"Split ratios must sum to a positive value, got: {raw}")
    return tuple(x / total for x in values)


def random_split(df, out_dir, seed=42, ratios=(0.8,0.1,0.1)):
    random.seed(seed)
    df_sh = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df_sh)
    n1 = int(n*ratios[0]); n2 = n1+int(n*ratios[1])
    train = df_sh.iloc[:n1]; val = df_sh.iloc[n1:n2]; test = df_sh.iloc[n2:]
    train.to_csv(os.path.join(out_dir, "pairs_train.tsv"), sep='\t', index=False)
    val.to_csv(os.path.join(out_dir, "pairs_val.tsv"), sep='\t', index=False)
    test.to_csv(os.path.join(out_dir, "pairs_test.tsv"), sep='\t', index=False)
    print("Splits saved: train/val/test =", len(train), len(val), len(test))

def taxa_split(df, taxonomy_parquet, out_dir, holdout_genus_frac=0.1, seed=42):
    """
    按属(genus)划分 train/val/test
    - 保证 genus 只出现在 train 或 test，不会泄露
    - 自动处理 genus 数量太少导致的抽样错误
    """
    import random
    import pandas as pd

    tax = pd.read_parquet(taxonomy_parquet)
    tax['taxid'] = tax['taxid'].astype(str)
    tax_map = tax.set_index('taxid').to_dict(orient='index')

    # 构造 parent_map 方便向上爬
    parent_map = {}
    for t, rec in tax_map.items():
        parent_map[t] = str(rec.get('parent', '')) if rec.get('parent', '') is not None else ''

    def find_rank_species_to_genus(spec_taxid):
        cur = str(spec_taxid)
        visited = set()
        while True:
            rec = tax_map.get(cur)
            if rec is None:
                return None
            if str(rec.get('rank', '')).lower() == 'genus':
                return cur
            parent = str(rec.get('parent', ''))
            if not parent or parent in visited or parent == cur:
                return None
            visited.add(parent)
            cur = parent

    # 映射 species_taxid -> genus_taxid
    df['host_species_taxid'] = df['host_species_taxid'].astype(str)
    df['genus_taxid'] = df['host_species_taxid'].map(find_rank_species_to_genus)

    genus_counts = df['genus_taxid'].value_counts(dropna=True)
    genus_list = genus_counts.index.tolist()
    print(f"找到 {len(genus_list)} 个 genus, 样本分布前10:")
    print(genus_counts.head(10))

    if len(genus_list) == 0:
        raise RuntimeError("没有找到任何 genus_taxid，请检查 host_species_taxid 是否正确或 taxonomy_parquet 是否匹配")

    random.seed(seed)
    holdout_n = int(len(genus_list) * holdout_genus_frac)
    holdout_n = min(holdout_n, len(genus_list))  # 防止超范围
    if holdout_n == 0:
        holdout_n = 1  # 至少保留一个 genus 做测试

    holdout_genera = set(random.sample(genus_list, holdout_n))
    print(f"Holdout genera count: {holdout_n}")

    # 划分 train/test
    train = df[~df['genus_taxid'].isin(holdout_genera)]
    test = df[df['genus_taxid'].isin(holdout_genera)]

    # 再对 train 划分 val
    tr_sh = train.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(tr_sh)
    nval = max(1, int(n * 0.1)) if n > 1 else 0
    train_final = tr_sh.iloc[:-nval] if nval > 0 else tr_sh
    val_final = tr_sh.iloc[-nval:] if nval > 0 else pd.DataFrame(columns=tr_sh.columns)

    train_final.to_csv(os.path.join(out_dir, "pairs_train_taxa.tsv"), sep='\t', index=False)
    val_final.to_csv(os.path.join(out_dir, "pairs_val_taxa.tsv"), sep='\t', index=False)
    test.to_csv(os.path.join(out_dir, "pairs_test_taxa.tsv"), sep='\t', index=False)

    print("Taxa-split saved:")
    print(f"  train={len(train_final)} val={len(val_final)} test={len(test)}")
    print(f"  test中包含的genus数: {len(holdout_genera)}")


def main():
    
    p = argparse.ArgumentParser()
    p.add_argument("--raw_pairs", required=True)
    p.add_argument("--host_catalog", required=True)
    p.add_argument("--taxonomy_parquet", required=True)
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--split", choices=['random','taxa'], default='random')
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ratios", default="0.8,0.1,0.1")
    args = p.parse_args()

    df = build_pairs(args.raw_pairs, args.host_catalog, args.taxonomy_parquet, args.out_dir)
    if args.split == 'random':
        random_split(df, args.out_dir, seed=args.seed, ratios=parse_ratios(args.ratios))
    else:
        taxa_split(df, args.taxonomy_parquet, args.out_dir, holdout_genus_frac=0.1, seed=args.seed)
    
if __name__ == "__main__":
    main()
