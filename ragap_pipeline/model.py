"""GATv2 model for phage-host interaction prediction."""

from __future__ import annotations

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
