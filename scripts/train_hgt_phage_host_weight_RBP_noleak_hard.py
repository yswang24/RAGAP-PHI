import argparse
import time
import math
import json
from collections import defaultdict
import logging
import os
import csv
import typing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv,GATv2Conv,HeteroConv
from torch_geometric.loader import LinkNeighborLoader
import torch


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def configure_torch_runtime(
    *,
    enable_tf32: bool,
    deterministic: bool,
    cudnn_benchmark: bool,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark and not deterministic)
    if hasattr(torch, 'set_float32_matmul_precision'):
        try:
            torch.set_float32_matmul_precision('high' if enable_tf32 else 'highest')
        except Exception:
            pass


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to allowlist BaseStorage for torch.load safety (PyTorch >=2.6)
try:
    from torch_geometric.data.storage import BaseStorage
    import torch.serialization as _tser
    _tser.add_safe_globals([BaseStorage])
except Exception:
    pass

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_torch_load(path: str) -> tuple[HeteroData, typing.Union[dict, None]]:
    """
    Loads a .pt file. Accepts either:
      - torch.save((data, split_edge), path)
      - torch.save(data, path) where data is HeteroData
      - torch.save({'data':data, 'split_edge': split_edge}, path)
    Returns (data, split_edge_or_None)
    """
    obj = torch.load(path, weights_only=False, map_location='cpu')
    if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[0], HeteroData):
        return obj[0], obj[1]
    if isinstance(obj, HeteroData):
        return obj, None
    if isinstance(obj, dict) and 'data' in obj and isinstance(obj['data'], HeteroData):
        return obj['data'], obj.get('split_edge', None)
    raise RuntimeError("Unsupported .pt content. Please save torch.save((data, split_edge), path) or torch.save(data, path).")

def select_phage_host_relation(data: HeteroData):
    preferred = ('phage', 'infects', 'host')
    if preferred in data.edge_types:
        return preferred
    return None


def add_train_only_infects_relations(
    data: HeteroData,
    relation: tuple[str, str, str],
    train_edge_index: torch.Tensor,
) -> tuple[HeteroData, tuple[str, str, str]]:
    if relation != ('phage', 'infects', 'host'):
        raise RuntimeError(f"Unexpected phage-host supervision relation: {relation}")
    data[relation].edge_index = train_edge_index.contiguous()
    reverse_relation = ('host', 'infected_by', 'phage')
    data[reverse_relation].edge_index = train_edge_index[[1, 0], :].contiguous()
    return data, reverse_relation


def build_positive_host_map(
    src_idx: torch.Tensor,
    dst_idx: torch.Tensor,
) -> dict[int, list[int]]:
    pos_map: defaultdict[int, set[int]] = defaultdict(set)
    for src, dst in zip(src_idx.tolist(), dst_idx.tolist()):
        pos_map[int(src)].add(int(dst))
    return {key: sorted(value) for key, value in pos_map.items()}


def node_global_ids(batch: HeteroData, node_type: str, device: torch.device) -> torch.Tensor:
    store = batch[node_type]
    if hasattr(store, 'id') and store.id is not None:
        return store.id.to(device=device, dtype=torch.long)
    if hasattr(store, 'n_id') and store.n_id is not None:
        return store.n_id.to(device=device, dtype=torch.long)
    raise RuntimeError(f"{node_type} batch is missing both .id and .n_id")


def encode_anchor_embeddings(
    model: "GATv2MiniModel",
    node_type: str,
    x: torch.Tensor,
) -> torch.Tensor:
    hidden = F.relu(model.input_proj[node_type](x))
    return F.normalize(model.final_proj[node_type](hidden), p=2, dim=-1)

def find_phage_host_splits(data: HeteroData, ext_splits: typing.Union[dict, None]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """
    Return three pairs: (train_src, train_dst), (val_src, val_dst), (test_src, test_dst)
    as 1D cpu LongTensors.
    """
    # if provided externally (from saved tuple)
    if ext_splits is not None:
        def as_pair(x):
            if isinstance(x, torch.Tensor) and x.dim() == 2 and x.size(0) == 2:
                return x[0].cpu(), x[1].cpu()
            raise RuntimeError("External split format invalid")
        try:
            return as_pair(ext_splits['train']['edge']), as_pair(ext_splits['val']['edge']), as_pair(ext_splits['test']['edge'])
        except Exception:
            pass

    # find phage->host relation name
    rel = select_phage_host_relation(data)
    if rel is None:
        raise RuntimeError("No ('phage',*, 'host') relation in data.edge_types")

    rec = data[rel]
    # common attribute patterns
    patterns = [
        ('edge_index_train', 'edge_index_val', 'edge_index_test'),
        ('train_pos_edge_index', 'val_pos_edge_index', 'test_pos_edge_index'),
        ('train_edge_index', 'val_edge_index', 'test_edge_index'),
        ('train', 'val', 'test'),
    ]
    for a, b, c in patterns:
        if hasattr(rec, a) and hasattr(rec, b) and hasattr(rec, c):
            A = getattr(rec, a); B = getattr(rec, b); C = getattr(rec, c)
            if isinstance(A, torch.Tensor) and A.dim() == 2 and A.size(0) == 2:
                return (A[0].cpu(), A[1].cpu()), (B[0].cpu(), B[1].cpu()), (C[0].cpu(), C[1].cpu())

    # fallback: maybe top-level attribute data.split_edge
    if hasattr(data, 'split_edge'):
        se = getattr(data, 'split_edge')
        if isinstance(se, dict) and 'train' in se and 'val' in se and 'test' in se:
            def pair(e):
                if isinstance(e, torch.Tensor) and e.dim() == 2 and e.size(0) == 2:
                    return e[0].cpu(), e[1].cpu()
                raise RuntimeError("split_edge format invalid")
            return pair(se['train']['edge']), pair(se['val']['edge']), pair(se['test']['edge'])

    raise RuntimeError("Cannot find phage-host splits inside data; please save splits or provide as ext_splits.")

# -------------------------
# Data Inspection and Fixing
# -------------------------
def check_node_counts(data: HeteroData) -> dict[str, typing.Union[int, None]]:
    node_counts = {}
    for ntype in data.node_types:
        if hasattr(data[ntype], "num_nodes"):
            n_nodes = int(data[ntype].num_nodes)
        elif 'x' in data[ntype]:
            n_nodes = int(data[ntype].x.size(0))
        else:
            n_nodes = None
        node_counts[ntype] = n_nodes
    return node_counts

def check_edge_bounds(data: HeteroData, node_counts: dict[str, typing.Union[int, None]]) -> list[tuple]:
    bad_items = []
    for etype, eidx in data.edge_index_dict.items():
        if eidx is None or eidx.numel() == 0:
            continue
        e_cpu = eidx.cpu()
        if e_cpu.dim() != 2 or e_cpu.size(0) != 2:
            continue
        src_max = int(e_cpu[0].max()); src_min = int(e_cpu[0].min())
        dst_max = int(e_cpu[1].max()); dst_min = int(e_cpu[1].min())
        src_type, _, dst_type = etype
        src_n = node_counts.get(src_type)
        dst_n = node_counts.get(dst_type)
        if src_n is not None and (src_min < 0 or src_max >= src_n):  # Assume non-negative indices
            bad_items.append(("edge_index", etype, "src", src_min, src_max, src_n))
        if dst_n is not None and (dst_min < 0 or dst_max >= dst_n):
            bad_items.append(("edge_index", etype, "dst", dst_min, dst_max, dst_n))
    return bad_items

def check_split_bounds(name: str, s_cpu: torch.Tensor, d_cpu: torch.Tensor, src_n: typing.Union[int, None], dst_n: typing.Union[int, None]) -> list[tuple]:
    if s_cpu.numel() == 0:
        return []
    s_arr = s_cpu.numpy()
    d_arr = d_cpu.numpy()
    smin, smax = int(s_arr.min()), int(s_arr.max())
    dmin, dmax = int(d_arr.min()), int(d_arr.max())
    bad = []
    if src_n is not None and (smin < 0 or smax >= src_n):
        bad.append((name, 'phage', smin, smax, src_n))
    if dst_n is not None and (dmin < 0 or dmax >= dst_n):
        bad.append((name, 'host', dmin, dmax, dst_n))
    return bad

def save_invalid_examples(out_dir: str, name: str, s_arr: np.ndarray, d_arr: np.ndarray, src_n: int, dst_n: int, limit: int = 200):
    invalid = []
    for i, (si, di) in enumerate(zip(s_arr.tolist(), d_arr.tolist())):
        if not (0 <= si < src_n and 0 <= di < dst_n):
            invalid.append((i, int(si), int(di)))
        if len(invalid) >= limit:
            break
    if invalid:
        fn = os.path.join(out_dir, f"invalid_{name}_examples.tsv")
        with open(fn, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["idx", "src_idx", "dst_idx"])
            writer.writerows(invalid)
        logger.info(f"Saved {len(invalid)} invalid {name} examples to {fn}")

def filter_pairs(s_cpu: torch.Tensor, d_cpu: torch.Tensor, src_n: int, dst_n: int) -> tuple[torch.Tensor, torch.Tensor]:
    s_list, d_list = [], []
    for si, di in zip(s_cpu.tolist(), d_cpu.tolist()):
        if 0 <= si < src_n and 0 <= di < dst_n:
            s_list.append(int(si))
            d_list.append(int(di))
    return torch.tensor(s_list, dtype=torch.long), torch.tensor(d_list, dtype=torch.long)

def add_placeholder_ids(data: HeteroData, node_counts: dict[str, typing.Union[int, None]]):
    for ntype in ['phage', 'host']:
        if not hasattr(data[ntype], 'id') and node_counts.get(ntype) is not None:
            n = node_counts[ntype]
            logger.info(f"Adding numeric placeholder {ntype}.id = arange({n}) (dtype=int64)")
            data[ntype].id = torch.arange(n, dtype=torch.long)

def inspect_and_fix_data(
    data: HeteroData,
    train_src_cpu: torch.Tensor,
    train_dst_cpu: torch.Tensor,
    val_src_cpu: torch.Tensor,
    val_dst_cpu: torch.Tensor,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    fix_enable: bool = True,
    out_dir: str = "debug_out"
) -> tuple[HeteroData, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    os.makedirs(out_dir, exist_ok=True)
    node_counts = check_node_counts(data)
    bad_items = check_edge_bounds(data, node_counts)

    phage_n = node_counts.get('phage')
    host_n = node_counts.get('host')

    bad_splits = []
    bad_splits += check_split_bounds("train", train_src_cpu, train_dst_cpu, phage_n, host_n)
    bad_splits += check_split_bounds("val", val_src_cpu, val_dst_cpu, phage_n, host_n)
    bad_splits += check_split_bounds("test", test_src_cpu, test_dst_cpu, phage_n, host_n)

    bad_items += bad_splits

    if bad_items:
        logger.warning("Found out-of-bounds items:")
        for b in bad_items:
            logger.warning(str(b))
        with open(os.path.join(out_dir, "bad_items.txt"), "w", encoding="utf-8") as fo:
            for b in bad_items:
                fo.write(str(b) + "\n")

    if bad_splits and phage_n is not None and host_n is not None:
        save_invalid_examples(out_dir, "train", train_src_cpu.numpy(), train_dst_cpu.numpy(), phage_n, host_n)
        save_invalid_examples(out_dir, "val", val_src_cpu.numpy(), val_dst_cpu.numpy(), phage_n, host_n)
        save_invalid_examples(out_dir, "test", test_src_cpu.numpy(), test_dst_cpu.numpy(), phage_n, host_n)

    if fix_enable and bad_splits and phage_n is not None and host_n is not None:
        logger.info("Auto-cleaning split edges (backing up originals to debug_out/)...")
        torch.save((train_src_cpu.clone(), train_dst_cpu.clone()), os.path.join(out_dir, "train_split_backup.pt"))
        torch.save((val_src_cpu.clone(), val_dst_cpu.clone()), os.path.join(out_dir, "val_split_backup.pt"))
        torch.save((test_src_cpu.clone(), test_dst_cpu.clone()), os.path.join(out_dir, "test_split_backup.pt"))

        train_src_cpu, train_dst_cpu = filter_pairs(train_src_cpu, train_dst_cpu, phage_n, host_n)
        val_src_cpu, val_dst_cpu = filter_pairs(val_src_cpu, val_dst_cpu, phage_n, host_n)
        test_src_cpu, test_dst_cpu = filter_pairs(test_src_cpu, test_dst_cpu, phage_n, host_n)

        logger.info("After filter train/val/test sizes: %d / %d / %d", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))
        torch.save((train_src_cpu, train_dst_cpu), os.path.join(out_dir, "train_split_fixed.pt"))
        torch.save((val_src_cpu, val_dst_cpu), os.path.join(out_dir, "val_split_fixed.pt"))
        torch.save((test_src_cpu, test_dst_cpu), os.path.join(out_dir, "test_split_fixed.pt"))

    add_placeholder_ids(data, node_counts)

    return data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu

# -------------------------
# Model
# -------------------------

import typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv


def etype_key(etype: tuple[str, str, str]) -> str:
    return f"{etype[0]}__{etype[1]}__{etype[2]}"


class RelationAttentionGATv2Layer(nn.Module):
    def __init__(
        self,
        node_types: tuple[str, ...],
        edge_types: tuple[tuple[str, str, str], ...],
        hidden_dim: int,
        n_heads: int,
        dropout: float,
        use_edge_attr: bool,
        edge_attr_dim: int,
    ):
        super().__init__()
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.use_edge_attr = use_edge_attr
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleDict()
        self.relation_gate = nn.ModuleDict()
        self.norm = nn.ModuleDict()

        for (src, rel, dst) in self.edge_types:
            conv_kwargs = {
                "in_channels": hidden_dim,
                "out_channels": hidden_dim,
                "heads": n_heads,
                "concat": False,
                "dropout": dropout,
                "add_self_loops": src == dst,
            }
            if use_edge_attr:
                conv_kwargs["edge_dim"] = edge_attr_dim
            self.convs[etype_key((src, rel, dst))] = GATv2Conv(**conv_kwargs)

        for node_type in self.node_types:
            self.relation_gate[node_type] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            self.norm[node_type] = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: typing.Optional[dict[tuple[str, str, str], torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        relation_outputs: dict[str, list[torch.Tensor]] = {node_type: [] for node_type in self.node_types}

        for etype, edge_index in edge_index_dict.items():
            if edge_index is None or edge_index.numel() == 0:
                continue
            src_type, _, dst_type = etype
            conv = self.convs[etype_key(etype)]
            conv_kwargs = {}
            if self.use_edge_attr and edge_attr_dict is not None and etype in edge_attr_dict:
                conv_kwargs["edge_attr"] = edge_attr_dict[etype]
            relation_outputs[dst_type].append(
                conv((h_dict[src_type], h_dict[dst_type]), edge_index, **conv_kwargs)
            )

        out_dict = {}
        for node_type in self.node_types:
            residual = h_dict[node_type]
            outputs = relation_outputs[node_type]
            if outputs:
                stacked = torch.stack(outputs, dim=1)
                attn_logits = self.relation_gate[node_type](stacked).squeeze(-1)
                attn_weights = torch.softmax(attn_logits, dim=1)
                aggregated = torch.sum(stacked * attn_weights.unsqueeze(-1), dim=1)
            else:
                aggregated = torch.zeros_like(residual)
            updated = self.norm[node_type](residual + self.dropout(aggregated))
            out_dict[node_type] = F.relu(updated)
        return out_dict


class GATv2MiniModel(nn.Module):
    def __init__(
        self,
        metadata: tuple,
        in_dims: dict,
        hidden_dim: int = 256,
        out_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        decoder: str = "mlp",
        use_edge_attr: bool = True,
        edge_attr_dim: int = 1,
        rel_init_map: typing.Optional[dict] = None,
        relation_aggr: str = "sum",
    ):
        super().__init__()
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.decoder_type = decoder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.dropout_p = dropout
        self.use_edge_attr = use_edge_attr
        self.edge_attr_dim = edge_attr_dim
        self.rel_init_map = rel_init_map
        self.relation_aggr = relation_aggr
        if self.relation_aggr not in {"sum", "attention"}:
            raise ValueError(f"Unsupported relation_aggr: {self.relation_aggr}")

        self.input_proj = nn.ModuleDict()
        for n in self.node_types:
            d = in_dims.get(n)
            if d is None:
                raise RuntimeError(f"Missing in_dim for node type {n}")
            self.input_proj[n] = nn.Linear(d, hidden_dim)

        concat_flag = False
        out_channels = hidden_dim

        self.edge_conv_md_list = nn.ModuleList()
        self.layers = nn.ModuleList()
        if self.relation_aggr == "attention":
            for _ in range(n_layers):
                self.layers.append(
                    RelationAttentionGATv2Layer(
                        tuple(self.node_types),
                        tuple(self.edge_types),
                        hidden_dim=hidden_dim,
                        n_heads=n_heads,
                        dropout=dropout,
                        use_edge_attr=use_edge_attr,
                        edge_attr_dim=edge_attr_dim,
                    )
                )
        else:
            for _ in range(n_layers):
                convs_md = nn.ModuleDict()
                for (src, rel, dst) in self.edge_types:
                    str_key = etype_key((src, rel, dst))
                    add_self_loops_flag = (src == dst)
                    if self.use_edge_attr:
                        conv = GATv2Conv(
                            in_channels=hidden_dim,
                            out_channels=out_channels,
                            heads=n_heads,
                            concat=concat_flag,
                            dropout=dropout,
                            edge_dim=self.edge_attr_dim,
                            add_self_loops=add_self_loops_flag,
                        )
                    else:
                        conv = GATv2Conv(
                            in_channels=hidden_dim,
                            out_channels=out_channels,
                            heads=n_heads,
                            concat=concat_flag,
                            dropout=dropout,
                            add_self_loops=add_self_loops_flag,
                        )
                    convs_md[str_key] = conv
                self.edge_conv_md_list.append(convs_md)
                conv_map = {et: convs_md[etype_key(et)] for et in self.edge_types}
                self.layers.append(HeteroConv(conv_map, aggr='sum'))

        self.dropout = nn.Dropout(self.dropout_p)
        self.final_proj = nn.ModuleDict({n: nn.Linear(hidden_dim, out_dim) for n in self.node_types})
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))
        if decoder == "mlp":
            self.decoder_mlp = self.edge_mlp

        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.rel_logw = nn.ParameterDict()
        rel_init_map = getattr(self, "rel_init_map", None)
        for etype in self.edge_types:
            if rel_init_map is not None and etype in rel_init_map:
                init_w = float(rel_init_map[etype])
            else:
                init_w = 1.0
            p = nn.Parameter(torch.log(torch.tensor(init_w, dtype=torch.float)))
            self.rel_logw[etype_key(etype)] = p

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
        edge_attr_dict: typing.Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        h = {n: F.relu(self.input_proj[n](x)) for n, x in x_dict.items()}
        for layer in self.layers:
            processed = None
            if self.use_edge_attr:
                processed = {}
                for etype, edge_index in edge_index_dict.items():
                    E = edge_index.size(1)
                    key = etype_key(etype)
                    alpha = torch.exp(self.rel_logw[key])
                    if edge_attr_dict is not None and etype in edge_attr_dict:
                        v = edge_attr_dict[etype]
                        if not torch.is_tensor(v):
                            v = torch.full(
                                (E, self.edge_attr_dim),
                                float(v),
                                dtype=torch.float,
                                device=edge_index.device,
                            )
                        else:
                            v = v.to(edge_index.device)
                            if v.dim() == 1:
                                v = v.view(-1, 1)
                            elif v.dim() == 2:
                                if v.size(1) != self.edge_attr_dim:
                                    if v.size(1) > self.edge_attr_dim:
                                        v = v[:, :self.edge_attr_dim]
                                    else:
                                        v = F.pad(v, (0, self.edge_attr_dim - v.size(1)), value=1.0)
                            else:
                                raise RuntimeError(f"edge_attr for {etype} must be 1D or 2D tensor")
                            if v.size(0) != E:
                                raise RuntimeError(
                                    f"edge_attr for {etype} length {v.size(0)} != edge count {E}"
                                )
                    else:
                        v = torch.ones((E, self.edge_attr_dim), device=edge_index.device)
                    processed[etype] = v * alpha
            if self.relation_aggr == "attention":
                h = layer(h, edge_index_dict, edge_attr_dict=processed)
            else:
                if self.use_edge_attr:
                    h = layer(h, edge_index_dict, edge_attr_dict=processed)
                else:
                    h = layer(h, edge_index_dict)
                for k in list(h.keys()):
                    h[k] = F.relu(self.dropout(h[k]))
        out = {k: F.normalize(self.final_proj[k](v), p=2, dim=-1) for k, v in h.items()}
        return out

    def decode(
        self,
        z_dict: dict[str, torch.Tensor],
        edge_label_index: typing.Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        etype: tuple[str, str, str]
    ) -> torch.Tensor:
        if isinstance(edge_label_index, torch.Tensor) and edge_label_index.dim() == 2 and edge_label_index.size(0) == 2:
            src_idx, dst_idx = edge_label_index[0], edge_label_index[1]
        elif isinstance(edge_label_index, (tuple, list)) and len(edge_label_index) == 2:
            src_idx, dst_idx = edge_label_index
        else:
            raise RuntimeError("edge_label_index must be (2,E) or tuple(src,dst)")

        src_type, _, dst_type = etype
        src_z = z_dict[src_type][src_idx]
        dst_z = z_dict[dst_type][dst_idx]

        if self.decoder_type == "cosine":
            src_n = F.normalize(src_z, p=2, dim=-1)
            dst_n = F.normalize(dst_z, p=2, dim=-1)
            sim = F.cosine_similarity(src_n, dst_n)
            return sim * torch.exp(self.logit_scale)
        elif self.decoder_type == "mlp":
            e = torch.cat([src_z, dst_z], dim=-1)
            return self.decoder_mlp(e).view(-1)
        else:
            raise ValueError(f"Unknown decoder {self.decoder_type}")
######全库评估
@torch.no_grad()
def compute_metrics_fullgraph(
    model: GATv2MiniModel,
    data: HeteroData,
    train_pairs: tuple[torch.Tensor, torch.Tensor],
    val_pairs: tuple[torch.Tensor, torch.Tensor],
    test_pairs: tuple[torch.Tensor, torch.Tensor],
    relation: tuple[str, str, str],
    eval_device: str = 'cpu',
    eval_neg_ratio: Optional[int] = 10,   # ← 允许 None,      # ← 允许 None
    k_list: tuple[int, ...] = (1, 5, 10, 20,30),
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    save_path: Optional[str] = None,
    top_k: int = 20,
    node_maps_path: str = "node_maps.json",
    edge_type_weight_map: typing.Optional[dict] = None
) -> tuple[tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]],
           tuple[float, float, dict[int, float]]]:

    from collections import defaultdict
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    import json
    import logging
    import math

    logger = logging.getLogger(__name__)

    # ======================
    # node id maps（与原逻辑一致）
    # ======================
    try:
        with open(node_maps_path, "r", encoding="utf-8") as f:
            node_maps = json.load(f)
        phage_map = node_maps.get("phage_map", {})
        host_map = node_maps.get("host_map", {})
        phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
        host_idx2id = {int(v): str(k) for k, v in host_map.items()}
    except FileNotFoundError:
        logger.warning("%s not found, using index as ID", node_maps_path)
        phage_idx2id = {}
        host_idx2id = {}

    # ======================
    # device
    # ======================
    orig_device = next(model.parameters()).device
    moved_model = False
    if orig_device != torch.device(eval_device):
        model.to(eval_device)
        moved_model = True

    try:
        data_eval_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        edge_index_dict_eval = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types
            if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }

        model.eval()

        # ======================
        # 边特征（统一二维形状）
        # ======================
        global_edge_attr = {}
        edge_type_weight_map = edge_type_weight_map or {}
        for et in data.edge_types:
            if et not in edge_index_dict_eval:
                continue
            if hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
                v = data[et].edge_weight.to(eval_device)
                global_edge_attr[et] = v.view(-1, 1)  # <- 统一 (E,1)
            elif et in edge_type_weight_map:
                E = data[et].edge_index.size(1)
                w = float(edge_type_weight_map[et])
                global_edge_attr[et] = torch.full((E, 1), w, device=eval_device)  # <- 统一 (E,1)
            else:
                # 不传该键，HeteroConv会走无 edge_attr 分支
                pass

        # forward 全图 embedding
        out = model(
            data_eval_x,
            edge_index_dict_eval,
            edge_attr_dict=global_edge_attr if len(global_edge_attr) > 0 else None
        )
        n_hosts = out['host'].size(0)

        # ======================
        # helpers（与原一致）
        # ======================
        def hostid2species(hid: int) -> str:
            if host_id2taxid is None or taxid2species is None:
                return f"unknown_{hid}"
            taxid = int(host_id2taxid[hid])
            return taxid2species.get(taxid, f"unknown_{taxid}")

        def build_pos_map(pairs: tuple[torch.Tensor, torch.Tensor]) -> defaultdict[int, set[int]]:
            pos_map = defaultdict(set)
            src_cpu, dst_cpu = pairs
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                pos_map[int(s)].add(int(d))
            return pos_map

        train_pos_map = build_pos_map(train_pairs)
        val_pos_map = build_pos_map(val_pairs)
        test_pos_map = build_pos_map(test_pairs)

        effective_top_k = max(top_k, max(k_list))

        # ======================
        # 全局正样本屏蔽表（严格 filtered setting）
        # ======================
        def build_global_pos_map(*pairs_list):
            gmap = defaultdict(set)
            for pairs in pairs_list:
                s_cpu, d_cpu = pairs
                for s, d in zip(s_cpu.tolist(), d_cpu.tolist()):
                    gmap[int(s)].add(int(d))
            return gmap

        global_pos_map = build_global_pos_map(train_pairs, val_pairs, test_pairs)

        # ======================
        # batched decode（用于全库评估时单个 phage 对候选 host 批打分）
        # ======================
        def batched_scores(model, z_dict, relation, phage_idx, host_idx_list, device, batch_size=4096):
            scores_all = []
            H = len(host_idx_list)
            for i in range(0, H, batch_size):
                hs = host_idx_list[i:i+batch_size]
                src = torch.full((len(hs),), phage_idx, dtype=torch.long, device=device)
                dst = torch.tensor(hs, dtype=torch.long, device=device)
                sc = model.decode(z_dict, (src, dst), etype=relation)  # (len(hs),)
                scores_all.append(sc.detach().cpu())
            return torch.cat(scores_all, dim=0)  # (H,)

        # ======================
        # AUC 计算（两种模式：采样 vs 全库）
        # ======================
        def compute_auc_for_pairs(pairs: tuple[torch.Tensor, torch.Tensor]) -> float:
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return float('nan')

            full_corpus_eval = (eval_neg_ratio is None) or (isinstance(eval_neg_ratio, int) and eval_neg_ratio < 0)

            pos_scores = []
            neg_scores_flat = []

            if getattr(model, "decoder_type", None) == "cosine":
                # 余弦解码的快路径
                host_emb = out['host']  # (H,D)
                logit_scale = torch.exp(model.logit_scale)

                # 逐条 pair 处理（便于严格屏蔽）
                for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
                    # 正例分数
                    pos = (out['phage'][s_idx:s_idx+1] @ host_emb[d_idx:d_idx+1].T) * logit_scale
                    pos = torch.sigmoid(pos).item()
                    pos_scores.append(pos)

                    # 构造候选负例
                    known = global_pos_map.get(s_idx, set())
                    mask = torch.ones(n_hosts, dtype=torch.bool)
                    for oth in known:
                        if 0 <= oth < n_hosts:
                            mask[oth] = False
                    if 0 <= d_idx < n_hosts:
                        mask[d_idx] = False

                    cand_neg_idx = mask.nonzero(as_tuple=True)[0]

                    if full_corpus_eval:
                        # 全库：用所有合法负例
                        if cand_neg_idx.numel() == 0:
                            continue
                        # 直接一批算完
                        s_vec = out['phage'][s_idx:s_idx+1]         # (1,D)
                        neg_emb = host_emb[cand_neg_idx]            # (N-,D)
                        neg = (s_vec @ neg_emb.T) * logit_scale     # (1,N-)
                        neg = torch.sigmoid(neg).view(-1).cpu().numpy()
                        neg_scores_flat.append(neg)
                    else:
                        # 采样：按 eval_neg_ratio
                        sample_size = min(int(eval_neg_ratio), int(cand_neg_idx.numel()))
                        if sample_size <= 0:
                            continue
                        perm = torch.randperm(cand_neg_idx.numel())[:sample_size]
                        neg_idx = cand_neg_idx[perm]
                        s_vec = out['phage'][s_idx:s_idx+1]
                        neg_emb = host_emb[neg_idx]
                        neg = (s_vec @ neg_emb.T) * logit_scale
                        neg = torch.sigmoid(neg).view(-1).cpu().numpy()
                        neg_scores_flat.append(neg)

            else:
                # 通用路径（解码器非cosine时）
                for s_idx, d_idx in zip(src_cpu.tolist(), dst_cpu.tolist()):
                    pos = model.decode(out, (torch.tensor([s_idx], device=eval_device),
                                             torch.tensor([d_idx], device=eval_device)), etype=relation)
                    pos = torch.sigmoid(pos).item()
                    pos_scores.append(pos)

                    known = global_pos_map.get(s_idx, set())
                    mask = torch.ones(n_hosts, dtype=torch.bool, device=eval_device)
                    for oth in known:
                        if 0 <= oth < n_hosts:
                            mask[oth] = False
                    if 0 <= d_idx < n_hosts:
                        mask[d_idx] = False

                    cand_neg_idx = mask.nonzero(as_tuple=True)[0]
                    if cand_neg_idx.numel() == 0:
                        continue

                    if (eval_neg_ratio is None) or (isinstance(eval_neg_ratio, int) and eval_neg_ratio < 0):
                        # 全库：全部负例
                        src = torch.full((cand_neg_idx.numel(),), s_idx, device=eval_device)
                        neg = model.decode(out, (src, cand_neg_idx), etype=relation)
                        neg = torch.sigmoid(neg).detach().cpu().numpy()
                        neg_scores_flat.append(neg)
                    else:
                        # 采样：eval_neg_ratio
                        sample_size = min(int(eval_neg_ratio), int(cand_neg_idx.numel()))
                        if sample_size <= 0:
                            continue
                        perm = torch.randperm(cand_neg_idx.numel(), device=eval_device)[:sample_size]
                        neg_idx = cand_neg_idx[perm]
                        src = torch.full((neg_idx.numel(),), s_idx, device=eval_device)
                        neg = model.decode(out, (src, neg_idx), etype=relation)
                        neg = torch.sigmoid(neg).detach().cpu().numpy()
                        neg_scores_flat.append(neg)

            if len(pos_scores) == 0 or len(neg_scores_flat) == 0:
                return float('nan')

            y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(sum(len(n) for n in neg_scores_flat))])
            y_score = np.concatenate([np.array(pos_scores), np.concatenate(neg_scores_flat)])
            try:
                return float(roc_auc_score(y_true, y_score))
            except Exception:
                return float('nan')

        # ======================
        # 排名指标（与原逻辑相同，但已严格屏蔽）
        # ======================
        def compute_rank_metrics(pairs: tuple[torch.Tensor, torch.Tensor], save_path: Optional[str] = None):
            src_cpu, dst_cpu = pairs
            if src_cpu.numel() == 0:
                return 0.0, {k: 0.0 for k in k_list}

            ph2hosts = defaultdict(list)
            for s, d in zip(src_cpu.tolist(), dst_cpu.tolist()):
                ph2hosts[int(s)].append(int(d))

            hits = {k: 0 for k in k_list}
            total_q = len(ph2hosts)
            prediction_rows = []
            rr_sum = 0.0

            if getattr(model, "decoder_type", None) == "cosine":
                ph_indices = list(ph2hosts.keys())
                if len(ph_indices) == 0:
                    return 0.0, {k: 0.0 for k in k_list}
                ph_emb = out['phage'][ph_indices]              # (P, D)
                host_emb = out['host']                         # (H, D)
                scores = torch.matmul(ph_emb, host_emb.t()) * torch.exp(model.logit_scale)
                scores_np = scores.cpu().numpy()

                for i_row, ph_global in enumerate(ph_indices):
                    row = scores_np[i_row].copy()

                    # 屏蔽该 phage 的所有其它真 host（train∪val∪test）
                    mask_set = set(global_pos_map.get(ph_global, set())) - set(ph2hosts[ph_global])
                    for hpos in mask_set:
                        if 0 <= hpos < row.shape[0]:
                            row[hpos] = -np.inf

                    K = min(effective_top_k, row.shape[0])
                    topk_idx = np.argpartition(-row, range(K))[:K]
                    topk_idx = topk_idx[np.argsort(-row[topk_idx])]

                    phage_real_id = phage_idx2id.get(ph_global, str(ph_global))
                    true_species = {hostid2species(h) for h in ph2hosts[ph_global]}

                    for rank_pos, h_idx in enumerate(topk_idx, 1):
                        host_real_id = host_idx2id.get(int(h_idx), str(int(h_idx)))
                        host_species_name = hostid2species(int(h_idx))
                        score_val = float(torch.sigmoid(torch.tensor(row[h_idx])).item())
                        prediction_rows.append({
                            "phage_id": phage_real_id,
                            "rank": rank_pos,
                            "host_id": host_real_id,
                            "host_species": host_species_name,
                            "score": score_val
                        })

                    # reciprocal rank（以 species 命中为准）
                    rank_val = None
                    for pos, h in enumerate(topk_idx, 1):
                        if hostid2species(int(h)) in true_species:
                            rank_val = pos
                            break
                    if rank_val is None:
                        rank_val = K + 1
                    rr_sum += 1.0 / rank_val

                    for k in k_list:
                        if any(hostid2species(int(h)) in true_species for h in topk_idx[:k]):
                            hits[k] += 1

                mrr = rr_sum / total_q if total_q > 0 else 0.0
                hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}

            else:
                for ph_idx, true_ds in ph2hosts.items():
                    scores_tensor = torch.sigmoid(
                        model.decode(out, (torch.full((n_hosts,), ph_idx, device=eval_device),
                                           torch.arange(n_hosts, device=eval_device)),
                                     etype=relation)
                    )
                    scores_np_all = scores_tensor.cpu().numpy()

                    row = scores_np_all.copy()
                    mask_set = set(global_pos_map.get(ph_idx, set())) - set(true_ds)
                    for hpos in mask_set:
                        if 0 <= hpos < row.shape[0]:
                            row[hpos] = -np.inf

                    K = min(effective_top_k, row.shape[0])
                    topk_idx = row.argsort()[::-1][:K]

                    phage_real_id = phage_idx2id.get(ph_idx, str(ph_idx))
                    true_species = {hostid2species(h) for h in true_ds}

                    for rank, h in enumerate(topk_idx, 1):
                        host_real_id = host_idx2id.get(int(h), str(int(h)))
                        host_species_name = hostid2species(int(h))
                        score = float(row[int(h)])
                        prediction_rows.append({
                            "phage_id": phage_real_id,
                            "rank": rank,
                            "host_id": host_real_id,
                            "host_species": host_species_name,
                            "score": score
                        })

                    rank_val = None
                    for pos, h in enumerate(topk_idx, 1):
                        if hostid2species(int(h)) in true_species:
                            rank_val = pos
                            break
                    if rank_val is None:
                        rank_val = K + 1
                    rr_sum += 1.0 / rank_val

                    for k in k_list:
                        if any(hostid2species(int(h)) in true_species for h in topk_idx[:k]):
                            hits[k] += 1

                mrr = rr_sum / total_q if total_q > 0 else 0.0
                hits_at = {k: hits[k] / total_q if total_q > 0 else 0.0 for k in k_list}

            if save_path is not None and len(prediction_rows) > 0:
                pd.DataFrame(prediction_rows).to_csv(save_path, sep="\t", index=False)

            return mrr, hits_at

        # ======================
        # 执行指标计算（内部已严格屏蔽；AUC分采样/全库）
        # ======================
        train_auc = compute_auc_for_pairs(train_pairs)
        train_mrr, train_hits = compute_rank_metrics(train_pairs, save_path=f"{save_path}_train_topk.tsv" if save_path else None)

        val_auc = compute_auc_for_pairs(val_pairs)
        val_mrr, val_hits = compute_rank_metrics(val_pairs, save_path=f"{save_path}_val_topk.tsv" if save_path else None)

        test_auc = compute_auc_for_pairs(test_pairs)
        test_mrr, test_hits = compute_rank_metrics(test_pairs, save_path=f"{save_path}_test_topk.tsv" if save_path else None)

        return (train_auc, train_mrr, train_hits), (val_auc, val_mrr, val_hits), (test_auc, test_mrr, test_hits)

    finally:
        if moved_model:
            model.to(orig_device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
def save_predictions(
    model: GATv2MiniModel,
    data: HeteroData,
    test_src_cpu: torch.Tensor,
    test_dst_cpu: torch.Tensor,
    relation: tuple[str, str, str],
    eval_device: torch.device,
    host_id2taxid: typing.Union[np.ndarray, None] = None,
    taxid2species: typing.Union[dict[int, str], None] = None,
    output_file: str = "phage_prediction_results.tsv",
    top_k: int = 20,  # control top-k in output
    k_list=(1, 5, 10,20,30),
    edge_type_weight_map: typing.Optional[dict] = None,
    edge_attr_dict: typing.Optional[dict] = None,
    node_maps_path: str = "node_maps.json"
 ):
    """
    保存 top-k 的 phage->host 预测（向量化 / 分批两种策略）。
    - 如果 model.decoder_type == "cosine": 使用一次性矩阵乘法计算 (num_phage, num_host) scores。
    - 否则：对每个 phage 分批调用 model.decode（避免 OOM）。
    """
    orig_dev = next(model.parameters()).device
    moved = False
    if orig_dev != eval_device:
        model.to(eval_device)
        moved = True

    try:
        # 全图节点特征与边索引（移动到 eval_device）
        full_x = {nt: data[nt].x.to(eval_device) for nt in data.node_types}
        full_edge_index_dict = {
            et: data[et].edge_index.to(eval_device)
            for et in data.edge_types if hasattr(data[et], 'edge_index') and data[et].edge_index is not None
        }

        # --- 构建 global_edge_attr (same logic as before) ---
        global_edge_attr = {}
        edge_type_weight_map = edge_type_weight_map or {}
        for et, edge_index in full_edge_index_dict.items():
            E = edge_index.size(1)
            if edge_attr_dict is not None and et in edge_attr_dict:
                val = edge_attr_dict[et]
                if isinstance(val, (float, int)):
                    global_edge_attr[et] = torch.full((E,), float(val), device=eval_device)
                elif isinstance(val, torch.Tensor):
                    t = val.to(eval_device)
                    if t.dim() == 0:
                        t = t.expand(E)
                    if t.dim() == 1 and t.size(0) != E:
                        raise RuntimeError(f"edge_attr for {et} has length {t.size(0)} != E({E})")
                    global_edge_attr[et] = t
                else:
                    raise RuntimeError(f"Unsupported edge_attr type for {et}: {type(val)}")
            elif hasattr(data[et], 'edge_weight') and data[et].edge_weight is not None:
                w = data[et].edge_weight.to(eval_device)
                if w.dim() == 0:
                    w = w.expand(E)
                elif w.dim() == 1 and w.size(0) != E:
                    raise RuntimeError(f"data[{et}].edge_weight length {w.size(0)} != E({E})")
                global_edge_attr[et] = w
            elif et in edge_type_weight_map:
                w = float(edge_type_weight_map[et])
                global_edge_attr[et] = torch.full((E,), w, device=eval_device)
            else:
                pass

        edge_attr_arg = global_edge_attr if len(global_edge_attr) > 0 else None

        # --- Forward (eval) to get embeddings for all nodes ---
        model.eval()
        with torch.no_grad():
            out_full = model(full_x, full_edge_index_dict, edge_attr_dict=edge_attr_arg)

            # all host embeddings and phage embeddings (full graph)
            phage_emb_all = out_full['phage']  # (N_phage, D)
            host_emb_all = out_full['host']    # (N_host, D)

            # Unique phage ids we need to output predictions for:
            # We will output per unique phage present in test_src_cpu
            uniq_phage_idx = torch.unique(test_src_cpu).to(eval_device)  # global phage indices (CPU -> move)
            # Map to int list for iteration / output ordering
            uniq_phage_list = uniq_phage_idx.cpu().tolist()

            # Load node maps
            try:
                with open(node_maps_path, "r", encoding="utf-8") as f:
                    node_maps = json.load(f)
                phage_map = node_maps.get("phage_map", {})
                host_map = node_maps.get("host_map", {})
                phage_idx2id = {int(v): str(k) for k, v in phage_map.items()}
                host_idx2id = {int(v): str(k) for k, v in host_map.items()}
            except Exception:
                logger = logging.getLogger(__name__)
                logger.warning("%s not found or invalid JSON, using index as ID", node_maps_path)
                phage_idx2id = {}
                host_idx2id = {}

            # host species mapping (if available)
            host_species_lookup = None
            if host_id2taxid is not None and taxid2species is not None:
                host_species_lookup = (host_id2taxid, taxid2species)

            phage2preds = {}

            if model.decoder_type == "cosine":
                # Vectorized scoring: (P_subset, D) @ (N_host, D).T -> (P_subset, N_host)
                # Use float32 on device; multiply by exp(logit_scale) to match decode behavior.
                scale = float(torch.exp(model.logit_scale).to(eval_device)) if hasattr(model, "logit_scale") else 1.0

                # To avoid very large memory, we can chunk phages (and/or hosts) if needed.
                # We'll iterate over phage chunks.
                ph_chunk = 256  # adjust if OOM; increase if memory allows
                N_host = host_emb_all.size(0)
                for i in range(0, len(uniq_phage_list), ph_chunk):
                    ph_idxs = uniq_phage_list[i:i+ph_chunk]
                    ph_tensor = phage_emb_all[torch.tensor(ph_idxs, device=eval_device)]  # (P_chunk, D)
                    # scores matrix:
                    scores_mat = torch.matmul(ph_tensor, host_emb_all.t()) * scale  # (P_chunk, N_host)
                    # convert to probabilities consistent with other code (you often used sigmoid)
                    probs_mat = torch.sigmoid(scores_mat)  # (P_chunk, N_host)

                    # get top-k per phage row
                    k = min(top_k, N_host)
                    top_vals, top_idxs = torch.topk(probs_mat, k=k, dim=1)
                    top_vals = top_vals.cpu().numpy()
                    top_idxs = top_idxs.cpu().numpy()

                    for r, pid in enumerate(ph_idxs):
                        ph_real = phage_idx2id.get(int(pid), str(int(pid)))
                        row_vals = top_vals[r]
                        row_idxs = top_idxs[r]
                        phage2preds[ph_real] = []
                        for rank_i, (hid_idx, score_val) in enumerate(zip(row_idxs.tolist(), row_vals.tolist()), start=1):
                            host_real = host_idx2id.get(int(hid_idx), str(int(hid_idx)))
                            host_sp = "NA"
                            if host_species_lookup is not None:
                                hid_global = int(hid_idx)
                                taxid_arr, taxmap = host_species_lookup
                                host_sp = taxmap.get(int(taxid_arr[hid_global]), "NA") if hid_global < len(taxid_arr) else "NA"
                            phage2preds[ph_real].append((host_real, host_sp, float(score_val)))

            else:
                # Non-cosine decoder: score each phage against all hosts but do it in batches
                # to avoid OOM; call model.decode on chunks of host indices.
                N_host = host_emb_all.size(0)
                host_idx_tensor_all = torch.arange(N_host, device=eval_device)
                host_chunk = 1024  # adjust
                for pid in uniq_phage_list:
                    ph_real = phage_idx2id.get(int(pid), str(int(pid)))
                    phage2preds[ph_real] = []
                    # compute scores against all hosts in chunks
                    for start in range(0, N_host, host_chunk):
                        end = min(N_host, start + host_chunk)
                        host_chunk_idx = host_idx_tensor_all[start:end]
                        # build edge_label_index for decode: (2, H_chunk) with phage repeated
                        ph_idx_tensor = torch.full((host_chunk_idx.numel(),), int(pid), dtype=torch.long, device=eval_device)
                        edge_label_index = (ph_idx_tensor, host_chunk_idx)
                        # model.decode expects z_dict (out_full) and edge_label_index
                        logits = model.decode(out_full, edge_label_index, etype=relation)  # (H_chunk,)
                        probs = torch.sigmoid(logits).cpu().numpy()
                        for hid_local, score_val in zip(host_chunk_idx.cpu().tolist(), probs.tolist()):
                            host_real = host_idx2id.get(int(hid_local), str(int(hid_local)))
                            host_sp = "NA"
                            if host_species_lookup is not None:
                                taxid_arr, taxmap = host_species_lookup
                                host_sp = taxmap.get(int(taxid_arr[int(hid_local)]), "NA") if int(hid_local) < len(taxid_arr) else "NA"
                            phage2preds[ph_real].append((host_real, host_sp, float(score_val)))
                    # after collecting all hosts, keep only top_k to save memory/disk
                    phage2preds[ph_real].sort(key=lambda x: x[2], reverse=True)
                    phage2preds[ph_real] = phage2preds[ph_real][:top_k]

        # --- write out top_k per phage ---
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["phage_id", "rank", "host_id", "host_species", "score"])
            for pid, preds in phage2preds.items():
                for rank, (hid, hs, s) in enumerate(preds, start=1):
                    writer.writerow([pid, rank, hid, hs, float(s)])

        logger = logging.getLogger(__name__)
        logger.info(f"Top-{top_k} Phage-host prediction results saved to {output_file}")

    finally:
        if moved:
            model.to(orig_dev)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# -------------------------
# Training entry
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_pt", required=True, help="Path to .pt file containing HeteroData or (data, split_edge)")
    p.add_argument("--taxid2species_tsv", default=None, help="Optional TSV mapping taxid -> species")
    p.add_argument("--device", default="cuda", help="training device")
    p.add_argument("--eval_device", default="cpu", help="eval device (use cpu to save GPU mem)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--out_dim", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_neighbors", nargs='+', type=int, default=[15,10], help="neighbors per hop, e.g. --num_neighbors 15 10")
    p.add_argument("--batch_size", type=int, default=2048, help="positive edges per batch")
    p.add_argument("--neg_ratio", type=int, default=1, help="compatibility alias for hard negatives when --hard_negatives is unset")
    p.add_argument("--hard_negatives", type=int, default=None, help="number of in-batch hard negatives per positive")
    p.add_argument("--tau", type=float, default=0.1, help="temperature used by the full-host softmax loss")
    p.add_argument("--eval_neg_ratio", type=int, default=1)
    p.add_argument("--save_path", default="best_hgt_nb.pt")
    p.add_argument("--log_every", type=int, default=None, help="compatibility alias for eval_every when --eval_every is unset")
    p.add_argument("--train_log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=None)
    p.add_argument("--loader_workers", type=int, default=8)
    p.add_argument("--pin_memory", type=str2bool, default=True)
    p.add_argument("--save_eval_predictions", type=str2bool, default=False)
    p.add_argument("--save_final_predictions", type=str2bool, default=True)
    p.add_argument("--enable_tf32", type=str2bool, default=True)
    p.add_argument("--deterministic", type=str2bool, default=False)
    p.add_argument("--cudnn_benchmark", type=str2bool, default=True)
    p.add_argument("--patience", type=int, default=6, help="number of evaluation points without val_mrr improvement before early stop")

    # NEW: node maps path (统一默认 node_maps.json)
    p.add_argument("--node_maps", default="node_maps_cluster_650.json", help="JSON file mapping node ids (phage_map, host_map). Default: node_maps.json")
    p.add_argument("--out_dir", default="outputs", help="directory to place all outputs (checkpoints, predictions, debug files)")
    p.add_argument("--host_cache_refresh_every", type=int, default=1, help="refresh the detached full-graph host cache every N epochs")
    p.add_argument("--relation_aggr", choices=["sum", "attention"], default="sum", help="cross-relation aggregation mode")
    p.add_argument("--train_objective", choices=["inbatch", "fullhost"], default="inbatch", help="training objective: in-batch CE or full-host softmax")
    p.add_argument("--message_passing_relation_scope", choices=["fullgraph", "trainonly"], default="fullgraph", help="whether phage-host message-passing edges include all splits or train edges only")
    return p.parse_args()

def bpr_loss(pos_scores, neg_scores):

    """
    BPR loss:
    pos_scores: [num_pos]
    neg_scores: [num_pos * neg_ratio] (flattened)
    """
    # 如果是一正多负 -> reshape
    num_pos = pos_scores.size(0)
    neg_ratio = neg_scores.size(0) // num_pos
    neg_scores = neg_scores.view(num_pos, neg_ratio)  # [num_pos, neg_ratio]

    # 广播比较正负
    diff = pos_scores.unsqueeze(1) - neg_scores  # [num_pos, neg_ratio]
    return -torch.mean(F.logsigmoid(diff))

def softmax_ce_loss(
    phage_emb_batch: torch.Tensor,
    host_emb_batch: torch.Tensor,
    pos_phage_local_idx: torch.LongTensor,
    pos_host_local_idx: torch.LongTensor,
    tau: float = 1.0,
    logit_scale: torch.Tensor = None,
) -> torch.Tensor:
    device = phage_emb_batch.device
    pos_phage_local_idx = pos_phage_local_idx.to(device)
    pos_host_local_idx = pos_host_local_idx.to(device)

    phage_vecs = phage_emb_batch[pos_phage_local_idx]
    logits = torch.matmul(phage_vecs, host_emb_batch.t())

    if logit_scale is not None:
        scale = torch.clamp(logit_scale, max=4.6).exp()
        logits = logits * scale
    else:
        logits = logits / tau

    labels = pos_host_local_idx.long()
    return F.cross_entropy(logits, labels, reduction='mean')

def multi_positive_full_host_softmax_loss(
    phage_emb_batch: torch.Tensor,
    host_emb_all: torch.Tensor,
    positive_mask: torch.Tensor,
    tau: float = 0.05,
    logit_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Full-host softmax with multi-positive masking.
    Each phage row is optimized against all host candidates, while multiple true
    hosts for the same phage share the positive probability mass.
    """
    if phage_emb_batch.dim() != 2 or host_emb_all.dim() != 2:
        raise RuntimeError('phage_emb_batch and host_emb_all must both be 2D tensors')
    if positive_mask.dim() != 2:
        raise RuntimeError('positive_mask must be a 2D tensor')
    if positive_mask.size(0) != phage_emb_batch.size(0):
        raise RuntimeError('positive_mask row count must match phage_emb_batch')
    if positive_mask.size(1) != host_emb_all.size(0):
        raise RuntimeError('positive_mask column count must match host_emb_all')
    if tau <= 0:
        raise RuntimeError('tau must be > 0')

    positive_mask = positive_mask.bool()
    valid_rows = positive_mask.any(dim=1)
    if not valid_rows.any():
        raise RuntimeError('multi-positive loss received no valid positive rows')
    if not valid_rows.all():
        phage_emb_batch = phage_emb_batch[valid_rows]
        positive_mask = positive_mask[valid_rows]

    logits = torch.matmul(phage_emb_batch, host_emb_all.t())
    if logit_scale is not None:
        logits = logits * torch.clamp(logit_scale, max=4.6).exp()
    else:
        logits = logits / tau

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive_logits = logits.masked_fill(~positive_mask, torch.finfo(logits.dtype).min)
    log_num = torch.logsumexp(positive_logits, dim=1)
    log_den = torch.logsumexp(logits, dim=1)
    return -(log_num - log_den).mean()


def build_edge_attr_dict_from_data(
    data: HeteroData,
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    device: torch.device,
    edge_type_weight_map: typing.Optional[dict[tuple[str, str, str], float]] = None,
) -> dict[tuple[str, str, str], torch.Tensor]:
    edge_type_weight_map = edge_type_weight_map or {}
    edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    for etype, edge_index in edge_index_dict.items():
        E = edge_index.size(1)
        if hasattr(data[etype], 'edge_weight') and data[etype].edge_weight is not None:
            edge_attr_dict[etype] = data[etype].edge_weight.to(device)
        elif etype in edge_type_weight_map:
            edge_attr_dict[etype] = torch.full((E,), float(edge_type_weight_map[etype]), device=device)
    return edge_attr_dict


@torch.no_grad()
def refresh_full_host_cache(
    model: GATv2MiniModel,
    data: HeteroData,
    compute_device: torch.device,
    target_device: torch.device,
    edge_type_weight_map: typing.Optional[dict[tuple[str, str, str], float]] = None,
) -> torch.Tensor:
    was_training = model.training
    original_device = next(model.parameters()).device
    compute_device = torch.device(compute_device)
    target_device = torch.device(target_device)

    if original_device != compute_device:
        model.to(compute_device)
    model.eval()

    full_x = {ntype: data[ntype].x.to(compute_device) for ntype in data.node_types}
    full_edge_index_dict = {
        etype: data[etype].edge_index.to(compute_device)
        for etype in data.edge_types
        if hasattr(data[etype], 'edge_index') and data[etype].edge_index is not None
    }
    full_edge_attr = build_edge_attr_dict_from_data(
        data,
        full_edge_index_dict,
        device=compute_device,
        edge_type_weight_map=edge_type_weight_map,
    )
    out_full = model(
        full_x,
        full_edge_index_dict,
        edge_attr_dict=full_edge_attr if full_edge_attr else None,
    )
    host_cache = out_full['host'].detach().to(target_device)

    del out_full
    del full_x
    del full_edge_index_dict
    del full_edge_attr

    if original_device != compute_device:
        model.to(original_device)
    if was_training:
        model.train()
    if original_device.type == 'cuda':
        torch.cuda.empty_cache()
    return host_cache


import pandas as pd
import random
import os
import torch
import logging
import time
import math
import argparse
from typing import Optional
from collections import defaultdict
from torch_geometric.loader import LinkNeighborLoader
import torch.nn.functional as F

def main():
    # ---------- 在 main() 里唯一定义 edge_type_weight_map ----------
    edge_type_weight_map = {
        ('phage', 'infects', 'host'): 3.0,
        # ('protein', 'similar', 'protein'): 0.8,
        # ('host', 'has_sequence', 'host_sequence'): 1.0,
        ('phage', 'interacts', 'phage'): 2.0,
        # ('host', 'interacts', 'host'): 1.0,
        # ('phage', 'encodes', 'protein'): 0.5,
        # ('host', 'encodes', 'protein'): 0.5,
        ('host', 'belongs_to', 'taxonomy'): 3.0,
        # ('taxonomy', 'related', 'taxonomy'): 2.5,
        # ('phage', 'belongs_to', 'taxonomy'): 1.0,
    }

    args = parse_args()
    if args.tau <= 0:
        raise ValueError("--tau must be > 0")
    effective_hard_negatives = args.hard_negatives if args.hard_negatives is not None else args.neg_ratio
    if effective_hard_negatives < 0:
        raise ValueError("--hard_negatives/--neg_ratio must be >= 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.train_log_every <= 0:
        raise ValueError("--train_log_every must be > 0")
    effective_eval_every = args.eval_every if args.eval_every is not None else args.log_every
    if effective_eval_every is None:
        effective_eval_every = 1000
    if effective_eval_every <= 0:
        raise ValueError("--eval_every/--log_every must be > 0")
    if args.loader_workers < 0:
        raise ValueError("--loader_workers must be >= 0")
    if args.host_cache_refresh_every <= 0:
        raise ValueError("--host_cache_refresh_every must be > 0")

    configure_torch_runtime(
        enable_tf32=args.enable_tf32,
        deterministic=args.deterministic,
        cudnn_benchmark=args.cudnn_benchmark,
    )

    # ---- create output dir and subdirs ----
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # subfolders
    debug_out_dir = os.path.join(out_dir, "debug_out")
    preds_dir = os.path.join(out_dir, "predictions")
    os.makedirs(debug_out_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # adjust save_path to be inside out_dir (preserve basename)
    args.save_path = os.path.join(out_dir, os.path.basename(args.save_path))

    # ---- add file logger into logging (besides console) ----
    # 配置 Logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除已有的 handlers 避免重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

    # 文件 Handler
    fh = logging.FileHandler(os.path.join(out_dir, "run.log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    eval_device = torch.device(args.eval_device)
    loader_num_workers = int(args.loader_workers)
    loader_pin_memory = bool(args.pin_memory and device.type == 'cuda')
    loader_persistent_workers = bool(loader_num_workers > 0)

    if torch.cuda.is_available():
        gpu_index = device.index if device.type == 'cuda' and device.index is not None else 0
        logger.info("Current GPU: %s", torch.cuda.get_device_name(gpu_index))
    logger.info(
        "Runtime flags: tf32=%s deterministic=%s cudnn_benchmark=%s eval_device=%s",
        args.enable_tf32,
        args.deterministic,
        args.cudnn_benchmark,
        eval_device,
    )

    logger.info("Loading data: %s", args.data_pt)
    data, split_edge = safe_torch_load(args.data_pt)
    logger.info("Data metadata: %s", data.metadata())

    train_pair, val_pair, test_pair = find_phage_host_splits(data, split_edge)
    train_src_cpu, train_dst_cpu = train_pair
    val_src_cpu, val_dst_cpu = val_pair
    test_src_cpu, test_dst_cpu = test_pair
    logger.info("Train/Val/Test counts: %d / %d / %d", train_src_cpu.size(0), val_src_cpu.size(0), test_src_cpu.size(0))

    data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu = inspect_and_fix_data(
        data, train_src_cpu, train_dst_cpu, val_src_cpu, val_dst_cpu, test_src_cpu, test_dst_cpu,
        fix_enable=True, out_dir=debug_out_dir
    )

    in_dims = {}
    for n in data.node_types:
        if 'x' not in data[n]:
            raise RuntimeError(f"Node {n} missing .x features")
        in_dims[n] = data[n].x.size(1)
        logger.info("node %s in_dim = %d", n, in_dims[n])

    relation = select_phage_host_relation(data)
    if relation is None:
        raise RuntimeError("phage->host relation not found")

    train_edge_index = torch.stack([train_src_cpu, train_dst_cpu], dim=0)
    if args.message_passing_relation_scope == "trainonly":
        data, _ = add_train_only_infects_relations(data, relation, train_edge_index)
        relation_edge_count = int(data[relation].edge_index.size(1)) if hasattr(data[relation], 'edge_index') and data[relation].edge_index is not None else 0
        logger.info(
            "Using train-only message-passing edges for %s=%d; val/test phage-host edges removed from encoder; train supervision edges=%d",
            relation,
            relation_edge_count,
            train_edge_index.size(1),
        )
    else:
        relation_edge_count = int(data[relation].edge_index.size(1)) if hasattr(data[relation], 'edge_index') and data[relation].edge_index is not None else 0
        logger.info(
            "Using original graph message-passing edges for %s=%d; train supervision edges=%d",
            relation,
            relation_edge_count,
            train_edge_index.size(1),
        )

    logger.info("Instantiating model...")

    model = GATv2MiniModel(
        metadata=data.metadata(), 
        in_dims=in_dims,
        hidden_dim=args.hidden_dim, 
        out_dim=args.out_dim,
        n_layers=args.n_layers, 
        n_heads=args.n_heads, 
        dropout=args.dropout, 
        decoder="cosine",
        use_edge_attr=True,
        edge_attr_dim=1,
        rel_init_map=edge_type_weight_map,
        relation_aggr=args.relation_aggr,
    ).to(device)

    optimizer = torch.optim.AdamW([
        {"params": [p for n,p in model.named_parameters() if ("logit_scale" not in n and "rel_logw" not in n)], "lr": args.lr},
        {"params": [model.logit_scale], "lr": args.lr * 0.1},
        {"params": list(model.rel_logw.parameters()), "lr": args.lr * 0.1}
    ], weight_decay=1e-5)
    
    train_positive_host_map = build_positive_host_map(train_src_cpu, train_dst_cpu)
    multi_positive_phages = sum(1 for hosts in train_positive_host_map.values() if len(hosts) > 1)
    max_hosts_per_phage = max((len(hosts) for hosts in train_positive_host_map.values()), default=0)
    train_objective_label = "full-host softmax CE" if args.train_objective == "fullhost" else "in-batch host softmax CE"
    host_cache = None
    host_cache_compute_device = torch.device(args.eval_device)

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors={etype: args.num_neighbors for etype in data.edge_types},
        edge_label_index=(relation, train_edge_index),
        edge_label=torch.ones(train_edge_index.size(1), dtype=torch.float),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        persistent_workers=loader_persistent_workers,
    )
    logger.info("Train loader created. batches: %d", len(train_loader))
    logger.info(
        "Training hyperparameters: tau=%.4f patience=%d num_neighbors=%s batch_size=%d hard_negatives_ignored=%d train_log_every=%d eval_every=%d",
        args.tau,
        args.patience,
        args.num_neighbors,
        args.batch_size,
        effective_hard_negatives,
        args.train_log_every,
        effective_eval_every,
    )
    logger.info(
        "Data loader: workers=%d pin_memory=%s persistent_workers=%s",
        loader_num_workers,
        loader_pin_memory,
        loader_persistent_workers,
    )
    logger.info(
        "Training objective: %s; relation_aggr=%s; multi-host phages=%d/%d max_hosts_per_phage=%d",
        train_objective_label,
        args.relation_aggr,
        multi_positive_phages,
        len(train_positive_host_map),
        max_hosts_per_phage,
    )

    taxid2species = None
    host_id2taxid = None
    if args.taxid2species_tsv:
        taxmap = pd.read_csv(args.taxid2species_tsv, sep="\t")
        taxid2species = dict(zip(taxmap["taxid"], taxmap["species"]))
        if hasattr(data['host'], 'taxid'):
            host_id2taxid = data['host'].taxid.cpu().numpy()
        else:
            logger.warning("data['host'] missing .taxid, species-level eval disabled")

    best_val_auc = -1.0
    best_val_mrr = -1.0
    best_epoch = 0
    evals_without_improve = 0
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        if args.train_objective == "fullhost" and (host_cache is None or (epoch - 1) % args.host_cache_refresh_every == 0):
            logger.info("[Epoch %03d] refreshing full-host cache on %s", epoch, host_cache_compute_device)
            host_cache = refresh_full_host_cache(
                model,
                data,
                compute_device=host_cache_compute_device,
                target_device=device,
                edge_type_weight_map=edge_type_weight_map,
            )
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # ================= Batch 循环开始 =================
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # 准备 x_dict 与 edge_index_dict
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            edge_index_dict = batch.edge_index_dict  # keys are tuples (src,rel,dst)

            # 构造 batch_edge_attr: 优先使用 batch 中已有的 edge_weight，否则使用你指定的 scalar map
            batch_edge_attr = {}
            for et in batch.edge_types:
                if hasattr(batch[et], 'edge_weight') and batch[et].edge_weight is not None:
                    v = batch[et].edge_weight.to(batch[et].edge_index.device)
                    batch_edge_attr[et] = v.view(-1, 1)  # 统一二维
                elif et in edge_type_weight_map:
                    # 有“类型先验”也不用在这里生成（由模型里的 rel_logw 学），
                    # 真要乘就把它当常数 per-edge 数值：
                    E = batch[et].edge_index.size(1)
                    batch_edge_attr[et] = torch.full((E,1), float(edge_type_weight_map[et]),
                                                    device=batch[et].edge_index.device)
            
            # 如果你希望“类型先验”只作为初始化，不想重复乘，就干脆不填 batch_edge_attr，让模型内部乘 alpha * 1。
            edge_attr_arg = batch_edge_attr if len(batch_edge_attr)>0 else None
            out = model(x_dict, edge_index_dict, edge_attr_dict=edge_attr_arg)

            # ========== 替换开始：用 in-batch softmax CE 代替原先的 BCE pos/neg ==========
            edge_label_index = batch[relation].edge_label_index
            if edge_label_index is None or edge_label_index.numel() == 0:
                continue

            # out 已由 model(...) 返回，为字典：out['phage'], out['host'], ...
            phage_emb_batch = out['phage']
            host_emb_batch = out['host']

            # positive pairs (local indices in this batch's phage/host sets)
            pos_src_local = edge_label_index[0].to(device)
            pos_dst_local = edge_label_index[1].to(device)
            if pos_src_local.numel() == 0:
                continue

            if args.train_objective == "fullhost":
                if host_cache is None:
                    raise RuntimeError("full-host training requested but host_cache is not initialized")
                batch_phage_global = node_global_ids(batch, relation[0], device)
                batch_host_global = node_global_ids(batch, relation[2], device)
                full_host_emb = host_cache.clone()
                full_host_emb[batch_host_global] = host_emb_batch
                pos_phage_global = batch_phage_global[pos_src_local]
                positive_mask = torch.zeros(
                    (pos_src_local.size(0), full_host_emb.size(0)),
                    dtype=torch.bool,
                    device=device,
                )
                for row_idx, phage_global in enumerate(pos_phage_global.tolist()):
                    positive_hosts = train_positive_host_map.get(int(phage_global))
                    if positive_hosts:
                        positive_mask[row_idx, positive_hosts] = True
                loss = multi_positive_full_host_softmax_loss(
                    phage_emb_batch[pos_src_local],
                    full_host_emb,
                    positive_mask,
                    tau=args.tau,
                    logit_scale=model.logit_scale,
                )
            else:
                loss = softmax_ce_loss(
                    phage_emb_batch,
                    host_emb_batch,
                    pos_src_local,
                    pos_dst_local,
                    tau=args.tau,
                    logit_scale=model.logit_scale,
                )
            
            # 反向传播
            loss.backward()
            
            # 可选：梯度裁剪，防止梯度爆炸（数值更稳）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # 更新参数
            optimizer.step()

            # 在更新后对 logit_scale 做 clamping，防止其变得过大/过小导致不稳定
            with torch.no_grad():
                # 关系门控 alpha = exp(logw)，这里clamp logw ∈ [ln 0.1, ln 5]
                for p in model.rel_logw.values():
                    p.clamp_(min=math.log(0.1), max=math.log(5.0))
                model.logit_scale.clamp_(-10.0, 10.0)

            # 清除梯度（你在循环开始也做了 zero_grad，这里在 step 后再清一次是安全的）
            optimizer.zero_grad()

            epoch_loss += float(loss.item())
            n_batches += 1
        # ================= Batch 循环结束 =================


        
        t1 = time.time()
        avg_loss = epoch_loss / max(1, n_batches)
        train_time_s = t1 - t0

        if epoch % args.train_log_every == 0 or epoch == args.epochs:
            logger.info(
                "[Epoch %03d] train_loss=%.6f train_time=%.1fs batches=%d",
                epoch,
                avg_loss,
                train_time_s,
                n_batches,
            )

        if epoch % effective_eval_every == 0 or epoch == args.epochs:
            try:
                eval_save_path = os.path.join(preds_dir, "phage_prediction_results") if args.save_eval_predictions else None
                train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
                    model, data, train_pair, val_pair, test_pair, relation=relation,
                    eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
                    host_id2taxid=host_id2taxid, taxid2species=taxid2species, k_list=(1, 5, 10, 20, 30),
                    save_path=eval_save_path, node_maps_path=args.node_maps,
                    edge_type_weight_map=edge_type_weight_map
                )
                train_auc, train_mrr, train_hits = train_metrics
                val_auc, val_mrr, val_hits = val_metrics
                test_auc, test_mrr, test_hits = test_metrics

                if args.save_eval_predictions:
                    pred_file = os.path.join(preds_dir, f"phage_prediction_results_epoch_{epoch}.tsv")
                    save_predictions(
                        model, data, test_src_cpu, test_dst_cpu, relation,
                        eval_device, host_id2taxid, taxid2species,
                        output_file=pred_file, k_list=(30,),
                        edge_type_weight_map=edge_type_weight_map,
                        node_maps_path=args.node_maps
                    )

            except Exception as e:
                logger.warning("Full-graph eval failed: %s", e)
                import traceback
                logger.warning(traceback.format_exc())
                train_auc = val_auc = test_auc = float('nan')
                train_mrr = val_mrr = test_mrr = 0.0
                train_hits = val_hits = test_hits = {k: 0.0 for k in (1, 5, 10)}

            logger.info(
                "[Eval %03d] val_auc=%.4f val_mrr=%.4f hits@1/5/10=%s",
                epoch,
                val_auc,
                val_mrr,
                val_hits,
            )

            if val_mrr > best_val_mrr:
                best_val_auc = val_auc
                best_val_mrr = val_mrr
                best_epoch = epoch
                evals_without_improve = 0
                best_ckpt = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_auc': val_auc,
                    'val_mrr': val_mrr
                }
                torch.save(best_ckpt, args.save_path)
                logger.info("Saved best model -> %s", args.save_path)
            else:
                evals_without_improve += 1
                if args.patience > 0 and evals_without_improve >= args.patience:
                    logger.info(
                        "Early stopping triggered at epoch %d after %d evals without val_mrr improvement; best_epoch=%d best_val_mrr=%.4f best_val_auc=%.4f",
                        epoch,
                        evals_without_improve,
                        best_epoch,
                        best_val_mrr,
                        best_val_auc,
                    )
                    break

    if best_ckpt is not None:
        model.load_state_dict(best_ckpt['model_state'])
    logger.info("Evaluating final test...")
    train_metrics, val_metrics, test_metrics = compute_metrics_fullgraph(
        model, data, train_pair, val_pair, test_pair, relation=relation,
        eval_device=args.eval_device, eval_neg_ratio=args.eval_neg_ratio,
        host_id2taxid=host_id2taxid, taxid2species=taxid2species, k_list=(1, 5, 10, 20, 30),
        save_path=None,
        node_maps_path=args.node_maps,
        edge_type_weight_map=edge_type_weight_map
    )
    logger.info("FINAL TEST metrics (AUC, MRR, Hits@1/5/10): %s", test_metrics)

    if args.save_final_predictions:
        final_pred_file = os.path.join(preds_dir, "phage_prediction_results_final.tsv")
        save_predictions(
            model, data, test_src_cpu, test_dst_cpu, relation,
            eval_device, host_id2taxid, taxid2species,
            output_file=final_pred_file, k_list=(30,),
            edge_type_weight_map=edge_type_weight_map,
            node_maps_path=args.node_maps
        )


if __name__ == "__main__":
    main()



'''
python train_hgt_phage_host_weight_RBP_noleak_hard.py \
  --data_pt artifacts/ragap_phi/graph/hetero_graph.pt \
  --node_maps artifacts/ragap_phi/graph/node_maps.json \
  --device cuda \
  --eval_device cpu \
  --epochs 5 \
  --hidden_dim 128 \
  --out_dim 128 \
  --n_layers 2 \
  --n_heads 1 \
  --num_neighbors 15 10 \
  --batch_size 256 \
  --neg_ratio 2 \
  --eval_neg_ratio 1 \
  --save_path best_hgt_nb.pt \
  --taxid2species_tsv data/metadata/taxid_species.tsv \
  --dropout 0.2 \
  --log_every 5 \
  --out_dir artifacts/ragap_phi/train/debug_run
'''
