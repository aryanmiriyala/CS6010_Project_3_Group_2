"""Utility to package mined artifacts/features into compressed archives for Git."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
Q1_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = Q1_ROOT / "artifacts"
FEATURES_DIR = Q1_ROOT / "features"
ARCHIVES_DIR = Q1_ROOT / "archives"


def discover_support_keys() -> list[str]:
    keys: set[str] = set()
    for seed_dir in ARTIFACTS_DIR.glob("seed_*"):
        for class_dir in seed_dir.glob("class_*"):
            for support_dir in class_dir.glob("support_*"):
                parts = support_dir.name.split("_")
                if len(parts) == 2:
                    keys.add(parts[1])
    for seed_dir in FEATURES_DIR.glob("seed_*"):
        for support_dir in seed_dir.glob("support_*"):
            parts = support_dir.name.split("_")
            if len(parts) == 2:
                keys.add(parts[1])
    return sorted(keys)


def zip_tree(root: Path, zip_handle: zipfile.ZipFile, arc_prefix: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        zip_handle.write(path, arcname=str(arc_prefix / relative))


def package_artifacts_for_support(seed: int, support_key: str) -> Path | None:
    ratio_dir = f"support_{support_key}"
    seed_dir = ARTIFACTS_DIR / f"seed_{seed}"
    if not seed_dir.exists():
        return None
    class_dirs = []
    for class_dir in sorted(seed_dir.glob("class_*")):
        support_dir = class_dir / ratio_dir
        if support_dir.exists():
            class_dirs.append((class_dir.name, support_dir))
    if not class_dirs:
        return None

    artifacts_root = ARCHIVES_DIR / "artifacts" / f"seed_{seed}"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    zip_path = artifacts_root / f"{ratio_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for class_name, support_dir in class_dirs:
            arc_prefix = Path("artifacts") / f"seed_{seed}" / class_name / ratio_dir
            zip_tree(support_dir, zf, arc_prefix)
    return zip_path


def package_features_for_support(seed: int, support_key: str) -> Path | None:
    ratio_dir = f"support_{support_key}"
    seed_dir = FEATURES_DIR / f"seed_{seed}"
    support_dir = seed_dir / ratio_dir
    if not support_dir.exists():
        return None

    features_root = ARCHIVES_DIR / "features" / f"seed_{seed}"
    features_root.mkdir(parents=True, exist_ok=True)
    zip_path = features_root / f"{ratio_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arc_prefix = Path("features") / f"seed_{seed}" / ratio_dir
        zip_tree(support_dir, zf, arc_prefix)
    return zip_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package mined artifacts/features into .zip archives for sharing."
    )
    parser.add_argument(
        "--support-ratios",
        nargs="*",
        help="Specific support ratios to package (e.g., 0.10 0.20). Defaults to every ratio found.",
    )
    parser.add_argument(
        "--skip-artifacts",
        action="store_true",
        help="Skip zipping the artifacts and only package features.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip zipping the features and only package artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_keys = args.support_ratios if args.support_ratios else discover_support_keys()
    support_keys = []
    for key in raw_keys:
        try:
            support_keys.append(f"{float(key):.2f}")
        except ValueError:
            support_keys.append(key)
    if not support_keys:
        raise SystemExit("No support ratios found to package. Run the mining/feature scripts first.")

    seed_dirs = sorted(p for p in ARTIFACTS_DIR.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        seed_dirs = sorted(p for p in FEATURES_DIR.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        raise SystemExit("No seed directories found in artifacts or features.")

    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping malformed seed directory {seed_dir}")
            continue
        for key in support_keys:
            print(f"\n=== Packaging seed {seed} support ratio {key} ===")
            if not args.skip_artifacts:
                artifact_zip = package_artifacts_for_support(seed, key)
                if artifact_zip:
                    print(f"Artifacts → {artifact_zip.relative_to(REPO_ROOT)}")
                else:
                    print("No artifacts found for this seed/ratio; skipping artifact archive.")
            if not args.skip_features:
                feature_zip = package_features_for_support(seed, key)
                if feature_zip:
                    print(f"Features → {feature_zip.relative_to(REPO_ROOT)}")
                else:
                    print("No features found for this seed/ratio; skipping feature archive.")


if __name__ == "__main__":
    main()
