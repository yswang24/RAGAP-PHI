#!/usr/bin/env python3
# dna_bert_batch_dir.py
"""
Batch DNA-BERT embedding:
  - Input: directory containing many .fasta files
  - Output: for each fasta -> one parquet file with rows (sequence_id, embedding)
  - Skip if output parquet already exists
  - Supports sliding windows, batching, optional reverse-complement averaging, fp16
"""

import os
import argparse
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import random
import math

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO

# -------------------------
# Utilities
# -------------------------
def setup_logging(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())  # also print to stdout

def rc_seq(seq: str) -> str:
    """Reverse complement"""
    trans = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(trans)[::-1]

def kmer_list(seq: str, k: int):
    """Return list of k-mer strings for sequence"""
    L = len(seq)
    if L < k:
        return [seq]  # short seq fallback
    return [seq[i:i+k] for i in range(0, L - k + 1)]

def windows_from_kmers(kmers, window_tokens, stride_tokens):
    """Generator of windows (each window is a list of k-mers)"""
    if len(kmers) <= window_tokens:
        yield kmers  # single window (no truncation)
        return
    i = 0
    while i + window_tokens <= len(kmers):
        yield kmers[i:i+window_tokens]
        i += stride_tokens
    # tail window to cover end
    if i < len(kmers):
        yield kmers[max(0, len(kmers)-window_tokens):len(kmers)]

def sample_windows_indices(n_windows, max_windows):
    """Sample evenly up to max_windows indices from 0..n_windows-1"""
    if max_windows is None or n_windows <= max_windows:
        return list(range(n_windows))
    # uniform sampling by intervals
    step = n_windows / max_windows
    idxs = [int(i*step) for i in range(max_windows)]
    idxs = list(dict.fromkeys([min(n_windows-1, x) for x in idxs]))  # unique & bounded
    return idxs

def mean_pool_from_hidden(hidden, attention_mask):
    # hidden: (batch, seq_len, hidden_dim)
    # attention_mask: (batch, seq_len)
    mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * mask, dim=1)
    lengths = torch.clamp(mask.sum(dim=1), min=1e-9)
    return (summed / lengths)

# -------------------------
# Core embedding logic
# -------------------------
def embed_sequence_windows(seq, tokenizer, model, device, k, window_tokens, stride_tokens,
                           batch_size, precision, rc_flag=False, max_windows=None):
    """
    For a given sequence string, compute a single sequence-level embedding:
      - build k-mer list
      - produce windows (list of k-mer lists)
      - optionally sample windows to cap max_windows
      - batch-tokenize windows (join k-mers by space) and run model
      - window-level mean pooling, then mean across windows
      - if rc_flag, compute rc embedding and average with forward
    Returns: numpy array (hidden_dim,)
    """
    seq = seq.replace('U', 'T').replace('\n', '').strip()
    kmers = kmer_list(seq, k)
    all_windows = list(windows_from_kmers(kmers, window_tokens, stride_tokens))
    if len(all_windows) == 0:
        return None

    # optional sampling to limit number of windows (for very long sequences)
    idxs = sample_windows_indices(len(all_windows), max_windows)
    sampled_windows = [all_windows[i] for i in idxs]

    def embed_windows(windows):
        window_embs = []
        # process in batches
        for i in range(0, len(windows), batch_size):
            batch = windows[i:i+batch_size]
            # join kmers into a string with spaces (DNABERT style)
            texts = [" ".join(w) for w in batch]
            tok = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=window_tokens+2)
            input_ids = tok['input_ids'].to(device)
            attention_mask = tok['attention_mask'].to(device)
            with torch.no_grad():
                if precision == 'fp16' and device.type == 'cuda':
                    # use autocast
                    with torch.cuda.amp.autocast():
                        out = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    out = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = out.last_hidden_state  # (B, T, H)
                pooled = mean_pool_from_hidden(hidden, attention_mask)  # (B, H)
                window_embs.append(pooled.cpu().numpy())
        if not window_embs:
            return None
        return np.vstack(window_embs)  # (n_windows, H)

    forward_embs = embed_windows(sampled_windows)
    if forward_embs is None:
        return None
    seq_emb = np.mean(forward_embs, axis=0)

    if rc_flag:
        seq_rc = rc_seq(seq)
        kmers_rc = kmer_list(seq_rc, k)
        windows_rc = list(windows_from_kmers(kmers_rc, window_tokens, stride_tokens))
        idxs_rc = sample_windows_indices(len(windows_rc), max_windows)
        sampled_rc = [windows_rc[i] for i in idxs_rc]
        rc_embs = embed_windows(sampled_rc)
        if rc_embs is not None:
            rc_seq_emb = np.mean(rc_embs, axis=0)
            seq_emb = (seq_emb + rc_seq_emb) / 2.0

    # L2 normalize
    norm = np.linalg.norm(seq_emb)
    if norm > 0:
        seq_emb = seq_emb / norm
    return seq_emb.astype(np.float32)

# -------------------------
# Process a single fasta file
# -------------------------
def process_fasta_file(fasta_path: Path, out_dir: Path, tokenizer, model, device, args):
    out_path = out_dir / (fasta_path.stem + ".parquet")
    if out_path.exists():
        logging.info(f"[SKIP] {fasta_path.name} -> {out_path.name} (exists)")
        return "skip"

    # read sequences in file (support multiple records, though usually one)
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        logging.warning(f"[WARN] {fasta_path.name} has no sequences")
        return "empty"

    rows = []
    for rec in records:
        seq_id = rec.id
        seq_str = str(rec.seq).upper()
        try:
            emb = embed_sequence_windows(seq_str, tokenizer, model, device,
                                         k=args.k,
                                         window_tokens=args.window_tokens,
                                         stride_tokens=args.stride_tokens,
                                         batch_size=args.batch_size,
                                         precision=args.precision,
                                         rc_flag=args.rc,
                                         max_windows=args.max_windows)
            if emb is None:
                logging.warning(f"[WARN] {fasta_path.name}:{seq_id} embedding None")
                continue
            rows.append({"sequence_id": seq_id, "embedding": emb.tolist(), "source_file": fasta_path.name})
        except Exception as e:
            logging.exception(f"[ERROR] {fasta_path.name}:{seq_id} failed: {e}")
            return "error"

    if not rows:
        logging.warning(f"[WARN] {fasta_path.name} produced no embeddings")
        return "no_output"

    # save as parquet (embedding column is a list of floats)
    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    logging.info(f"[OK] Saved {out_path.name} ({len(rows)} sequences)")
    return "ok"

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Batch DNA-BERT embedding from a directory of FASTA files")
    parser.add_argument("--fasta_dir", required=True, help="Input directory with .fasta/.fa/.fna files")
    parser.add_argument("--out_dir", required=True, help="Output directory for parquet files")
    parser.add_argument("--model", required=True, help="Pretrained DNABERT model path or HF name")
    parser.add_argument("--k", type=int, default=6, help="k-mer size (must match model)")
    parser.add_argument("--window_tokens", type=int, default=510, help="Number of k-mer tokens per window")
    parser.add_argument("--stride_tokens", type=int, default=510, help="Stride (tokens) between windows")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for model inference")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--rc", action="store_true", help="Compute reverse-complement embedding and average (flag)")
    parser.add_argument("--max_windows", type=int, default=None, help="Max windows to sample per sequence (None = all)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32","fp16"], help="Precision for inference")
    parser.add_argument("--log", type=str, default="dna_bert_batch.log", help="Log file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log)
    logging.info("=== DNA-BERT batch embedding started ===")
    logging.info(f"Model={args.model}, k={args.k}, window_tokens={args.window_tokens}, stride={args.stride_tokens}, batch={args.batch_size}, device={args.device}, rc={args.rc}, max_windows={args.max_windows}, precision={args.precision}")

    # device
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    logging.info(f"Using device: {device}")

    # load tokenizer & model
    logging.info("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True,trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model,trust_remote_code=True)
    model.to(device)
    model.eval()

    # list fasta files
    fasta_dir = Path(args.fasta_dir)
    fasta_files = sorted([p for p in fasta_dir.iterdir() if p.suffix.lower() in (".fasta", ".fa", ".fna")])
    if not fasta_files:
        logging.error(f"No fasta files found in {fasta_dir}")
        return

    # process files
    summary = {"ok":0, "skip":0, "error":0, "empty":0, "no_output":0}
    for fasta_path in tqdm(fasta_files, desc="Files"):
        status = process_fasta_file(fasta_path, out_dir, tokenizer, model, device, args)
        if status in summary:
            summary[status] += 1
        else:
            summary["error"] += 1

    logging.info(f"Finished. summary: {summary}")
    logging.info("=== Done ===")

if __name__ == "__main__":
    main()

'''
python dna_bert_embed.py \
  --fasta_dir inputs/phage_fasta \
  --model assets/models/DNA_bert_4 \
  --out_dir artifacts/ragap_phi/dna/phage_embeddings \
  --k 4 \
  --window_tokens 510 \
  --stride_tokens 510 \
  --batch_size 32 \
  --device cuda \
  --precision fp16 \
  --max_windows 800 \
  --log dna_bert_batch.log

  

python dna_bert_embed.py \
  --fasta_dir inputs/host_fasta \
  --model assets/models/DNA_bert_4 \
  --out_dir artifacts/ragap_phi/dna/host_embeddings \
  --k 4 \
  --window_tokens 510 \
  --stride_tokens 510 \
  --batch_size 32 \
  --device cuda \
  --precision fp16 \
  --max_windows 800 \
  --log dna_bert_batch.log
'''
