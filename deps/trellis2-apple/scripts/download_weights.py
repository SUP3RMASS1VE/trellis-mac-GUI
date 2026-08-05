"""Download TRELLIS.2 weights into a local directory structure.

The rest of the codebase already knows how to load TRELLIS.2 from a local
snapshot directory as long as it contains the same layout as the Hugging Face
repo root:

    pipeline.json
    texturing_pipeline.json
    ckpts/*.json
    ckpts/*.safetensors

Example:
    python export/download_weights.py --output-dir weights/TRELLIS.2-4B
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


DEFAULT_PATTERNS = [
    "pipeline.json",
    "texturing_pipeline.json",
    "ckpts/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TRELLIS.2 weights locally")
    parser.add_argument("--repo-id", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--output-dir", default="weights/TRELLIS.2-4B")
    parser.add_argument(
        "--full-snapshot",
        action="store_true",
        help="Download the entire repo instead of just pipeline configs and checkpoints",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    download_kwargs = {
        "repo_id": args.repo_id,
        "local_dir": output_dir,
        "local_dir_use_symlinks": False,
        "resume_download": True,
    }
    if not args.full_snapshot:
        download_kwargs["allow_patterns"] = DEFAULT_PATTERNS

    snapshot_path = snapshot_download(**download_kwargs)
    print(f"Downloaded {args.repo_id} to {snapshot_path}")


if __name__ == "__main__":
    main()
