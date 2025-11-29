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
    for class_dir in ARTIFACTS_DIR.glob("class_*"):
        if not class_dir.is_dir():
            continue
        for support_dir in class_dir.glob("support_*"):
            parts = support_dir.name.split("_")
            if len(parts) == 2:
                keys.add(parts[1])
    for support_dir in FEATURES_DIR.glob("support_*"):
        if support_dir.is_dir():
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


def package_artifacts_for_support(support_key: str) -> Path | None:
    ratio_dir = f"support_{support_key}"
    class_dirs = []
    for class_dir in sorted(ARTIFACTS_DIR.glob("class_*") ):
        support_dir = class_dir / ratio_dir
        if support_dir.exists():
            class_dirs.append((class_dir.name, support_dir))
    if not class_dirs:
        return None

    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ARCHIVES_DIR / f"artifacts_{ratio_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for class_name, support_dir in class_dirs:
            arc_prefix = Path("artifacts") / class_name / ratio_dir
            zip_tree(support_dir, zf, arc_prefix)
    return zip_path


def package_features_for_support(support_key: str) -> Path | None:
    ratio_dir = f"support_{support_key}"
    support_dir = FEATURES_DIR / ratio_dir
    if not support_dir.exists():
        return None

    ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = ARCHIVES_DIR / f"features_{ratio_dir}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arc_prefix = Path("features") / ratio_dir
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

    for key in support_keys:
        print(f"\n=== Packaging support ratio {key} ===")
        if not args.skip_artifacts:
            artifact_zip = package_artifacts_for_support(key)
            if artifact_zip:
                print(f"Artifacts → {artifact_zip.relative_to(REPO_ROOT)}")
            else:
                print("No artifacts found for this ratio; skipping artifact archive.")
        if not args.skip_features:
            feature_zip = package_features_for_support(key)
            if feature_zip:
                print(f"Features → {feature_zip.relative_to(REPO_ROOT)}")
            else:
                print("No features found for this ratio; skipping feature archive.")


if __name__ == "__main__":
    main()
