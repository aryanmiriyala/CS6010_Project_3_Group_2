"""
Utility entrypoint to download/cache the MUTAG dataset before running experiments.

Usage:
    python data_download/download_mutag.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_access.mutag import ensure_mutag_downloaded


def main() -> None:
    num_graphs = ensure_mutag_downloaded()
    print(f"Downloaded/verified MUTAG with {num_graphs} graphs in ./data/MUTAG")


if __name__ == "__main__":
    main()
