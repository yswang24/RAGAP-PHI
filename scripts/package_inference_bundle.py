#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ID = "ragap_phi"
DEFAULT_BUNDLE_CHECKPOINT = Path("train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt")
DEFAULT_SOURCE_CHECKPOINT_CANDIDATES = (
    Path("train/fullhost_v2/best_GAT_attn_fullhost_copymsg_v2.pt"),
    Path("train_attn_fullhost_copymsg_v2/best_GAT_attn_fullhost_copymsg_v2.pt"),
)

# Minimal inference-time assets required by the current code path.
MINIMAL_REQUIRED_ITEMS = {
    Path("graph/hetero_graph.pt"): Path("graph/hetero_graph.pt"),
    Path("graph/node_maps.json"): Path("graph/node_maps.json"),
    Path("catalogs/host_catalog.parquet"): Path("catalogs/host_catalog.parquet"),
    Path("cluster/sourmash/phage_phage/signatures"): Path("cluster/sourmash/phage_phage/signatures"),
}

OPTIONAL_ITEMS = {
    Path("manifests/train.json"): Path("manifests/train.json"),
}


@dataclass(frozen=True)
class CopiedItem:
    source: Path
    destination: Path
    required: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the minimal inference bundle for a GitHub/Release deployment."
    )
    parser.add_argument(
        "--source-artifacts",
        required=True,
        help="Source artifact root, for example old_repo/artifacts/ragap_test",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output directory that will receive artifacts/<dataset_id>/...",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help=f"Destination dataset id under artifacts/. Default: {DEFAULT_DATASET_ID}",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Explicit source checkpoint path. If omitted, common old layouts are probed.",
    )
    parser.add_argument(
        "--include-manifest",
        action="store_true",
        help="Also copy manifests/train.json as optional metadata.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="Optional .tar.gz output path for the packaged artifacts/<dataset_id> tree.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output dataset directory.",
    )
    return parser.parse_args()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def _copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        _copy_tree(source, destination)
        return
    _copy_file(source, destination)


def _resolve_checkpoint(source_root: Path, explicit: str | None) -> Path:
    if explicit:
        checkpoint_path = Path(explicit).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    for candidate in DEFAULT_SOURCE_CHECKPOINT_CANDIDATES:
        checkpoint_path = (source_root / candidate).resolve()
        if checkpoint_path.exists():
            return checkpoint_path

    raise FileNotFoundError(
        "Unable to locate a trained checkpoint under the source artifact root. "
        "Pass --checkpoint explicitly if your layout is different."
    )


def _write_bundle_metadata(bundle_root: Path, copied_items: list[CopiedItem]) -> None:
    metadata = {
        "bundle_type": "ragap_phi_minimal_inference_bundle",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_root": str(bundle_root),
        "files": [
            {
                "source": str(item.source),
                "destination": str(item.destination),
                "required": item.required,
            }
            for item in copied_items
        ],
        "notes": [
            "This bundle contains only the minimal cached artifacts needed by the current inference code path.",
            "Third-party dependencies such as DNABERT, Pharokka databases, and ESM2 weights are not bundled here.",
            "The per-query sourmash signature is generated at runtime; the packaged signatures directory is the cached training reference library.",
            "train.json is optional metadata and is excluded unless --include-manifest is requested.",
        ],
    }
    metadata_path = bundle_root / "INFERENCE_BUNDLE.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _make_archive(bundle_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(bundle_root, arcname=str(Path("artifacts") / bundle_root.name))


def package_inference_bundle(
    source_artifacts: Path,
    output_root: Path,
    *,
    dataset_id: str = DEFAULT_DATASET_ID,
    checkpoint: str | None = None,
    include_manifest: bool = False,
    archive: Path | None = None,
    overwrite: bool = False,
) -> Path:
    source_root = source_artifacts.expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source artifact root not found: {source_root}")

    destination_root = output_root.expanduser().resolve() / "artifacts" / dataset_id
    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {destination_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(destination_root)

    copied_items: list[CopiedItem] = []
    for source_relpath, destination_relpath in MINIMAL_REQUIRED_ITEMS.items():
        source_path = source_root / source_relpath
        if not source_path.exists():
            raise FileNotFoundError(f"Required inference asset missing: {source_path}")
        destination_path = destination_root / destination_relpath
        _copy_path(source_path, destination_path)
        copied_items.append(CopiedItem(source=source_path, destination=destination_path, required=True))

    checkpoint_source = _resolve_checkpoint(source_root, checkpoint)
    checkpoint_destination = destination_root / DEFAULT_BUNDLE_CHECKPOINT
    _copy_file(checkpoint_source, checkpoint_destination)
    copied_items.append(CopiedItem(source=checkpoint_source, destination=checkpoint_destination, required=True))

    if include_manifest:
        for source_relpath, destination_relpath in OPTIONAL_ITEMS.items():
            source_path = source_root / source_relpath
            if not source_path.exists():
                continue
            destination_path = destination_root / destination_relpath
            _copy_file(source_path, destination_path)
            copied_items.append(CopiedItem(source=source_path, destination=destination_path, required=False))

    _write_bundle_metadata(destination_root, copied_items)

    if archive is not None:
        _make_archive(destination_root, archive.expanduser().resolve())

    return destination_root


def main() -> int:
    args = parse_args()
    bundle_root = package_inference_bundle(
        Path(args.source_artifacts),
        Path(args.output_root),
        dataset_id=args.dataset_id,
        checkpoint=args.checkpoint,
        include_manifest=args.include_manifest,
        archive=Path(args.archive) if args.archive else None,
        overwrite=args.overwrite,
    )
    print(f"Packaged minimal inference bundle at {bundle_root}")
    if args.archive:
        print(f"Archive created at {Path(args.archive).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
