#!/usr/bin/env python3
"""Download the Fast-dLLM v2 7B weights into the project-relative model cache.

The repo tracks only our *edited* ``modeling.py`` (with the FOCUS token-skipping,
delayed KV cache, and DynamicCache paths). The weights, ``config.json``,
``configuration.py`` and the tokenizer are NOT tracked (29 GB) and are fetched
from the Hugging Face Hub by this script into the exact directory the eval /
benchmark scripts expect:

    <project_root>/models/models--Efficient-Large-Model--Fast_dLLM_v2_7B/
        snapshots/0661abf5f9f0ee338970d091052a26c8efa51974/

The revision is pinned so the snapshot lands at the same path our tracked
``modeling.py`` lives in, and ``modeling.py`` is excluded from the download so
the Hub's (unedited) version never overwrites our edits.

Usage:
    python scripts/download_model.py
    # or point the eval at the cache dir it creates:
    #   model_path=<that snapshot dir>   (see eval_script.sh / run_configs_parallel.sh)
"""
import os
import sys

REPO_ID = "Efficient-Large-Model/Fast_dLLM_v2_7B"
REVISION = "0661abf5f9f0ee338970d091052a26c8efa51974"

# Project root = parent of this scripts/ directory. Everything is relative to it,
# so the checkout works from any clone location.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit(
            "huggingface_hub is required. Install it with:\n"
            "    pip install huggingface_hub"
        )

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Downloading {REPO_ID}@{REVISION[:8]} into {CACHE_DIR}")
    print("(skipping modeling.py — the repo's edited version is kept)")

    snapshot_path = snapshot_download(
        repo_id=REPO_ID,
        revision=REVISION,
        cache_dir=CACHE_DIR,
        # Preserve our tracked, edited modeling.py: never pull the Hub copy over it.
        ignore_patterns=["modeling.py"],
    )

    modeling = os.path.join(snapshot_path, "modeling.py")
    print(f"\nDone. Snapshot at:\n    {snapshot_path}")
    if os.path.exists(modeling):
        print("Edited modeling.py present (FOCUS / delayed-cache / dynamic paths).")
    else:
        print(
            "WARNING: modeling.py is missing from the snapshot dir.\n"
            "         It should have come from the git checkout — verify the clone."
        )


if __name__ == "__main__":
    main()
