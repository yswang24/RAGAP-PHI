from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import cluster, dna, graph, pairs, proteins, train


@dataclass(frozen=True)
class StageSpec:
    name: str
    section_path: tuple[str, ...]
    default_deps: tuple[str, ...]
    description: str
    default_mode: str
    module: Any


@dataclass
class StageRunResult:
    command: list[str] | None = None
    notes: dict[str, Any] = field(default_factory=dict)


STAGES = {
    "dna_embed_phage": StageSpec(
        name="dna_embed_phage",
        section_path=("dna_embedding", "phage"),
        default_deps=(),
        description="Phage DNA embeddings",
        default_mode="script",
        module=dna,
    ),
    "dna_embed_host": StageSpec(
        name="dna_embed_host",
        section_path=("dna_embedding", "host"),
        default_deps=(),
        description="Host DNA embeddings",
        default_mode="script",
        module=dna,
    ),
    "build_catalogs": StageSpec(
        name="build_catalogs",
        section_path=("build_catalogs",),
        default_deps=("dna_embed_phage", "dna_embed_host"),
        description="Build phage and host catalogs",
        default_mode="internal",
        module=pairs,
    ),
    "build_pairs": StageSpec(
        name="build_pairs",
        section_path=("pairs",),
        default_deps=("build_catalogs",),
        description="Build train/val/test pairs",
        default_mode="script",
        module=pairs,
    ),
    "prepare_phage_proteins": StageSpec(
        name="prepare_phage_proteins",
        section_path=("phage_protein_prep",),
        default_deps=(),
        description="Prepare phage protein FAA files",
        default_mode="internal",
        module=proteins,
    ),
    "prepare_host_proteins": StageSpec(
        name="prepare_host_proteins",
        section_path=("host_protein_prep",),
        default_deps=(),
        description="Prepare host protein FAA files",
        default_mode="internal",
        module=proteins,
    ),
    "embed_phage_proteins": StageSpec(
        name="embed_phage_proteins",
        section_path=("phage_protein_embedding",),
        default_deps=("prepare_phage_proteins",),
        description="Embed phage proteins",
        default_mode="script",
        module=proteins,
    ),
    "embed_host_proteins": StageSpec(
        name="embed_host_proteins",
        section_path=("host_protein_embedding",),
        default_deps=("prepare_host_proteins",),
        description="Embed host proteins",
        default_mode="script",
        module=proteins,
    ),
    "build_cluster_assets": StageSpec(
        name="build_cluster_assets",
        section_path=("cluster_assets",),
        default_deps=("embed_phage_proteins", "embed_host_proteins"),
        description="Normalize cluster protein assets",
        default_mode="internal",
        module=cluster,
    ),
    "build_graph": StageSpec(
        name="build_graph",
        section_path=("graph",),
        default_deps=("build_catalogs", "build_pairs", "build_cluster_assets"),
        description="Build heterogeneous graph",
        default_mode="script",
        module=graph,
    ),
    "train": StageSpec(
        name="train",
        section_path=("train",),
        default_deps=("build_graph",),
        description="Train graph model",
        default_mode="script",
        module=train,
    ),
}
