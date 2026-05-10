#!/usr/bin/env python3
"""
fetch_cloud_models.py — Query Comfy Cloud for available model files.

Queries /api/models/<folder> for each model type (unet, clip, vae, checkpoints,
loras) and writes a snapshot to cloud-models.json, consumed by the workflow-audit
and comfy-cloud-models skills.

Usage:
    COMFY_CLOUD_API_KEY=xxx python fetch_cloud_models.py --output cloud-models.json
    python fetch_cloud_models.py --api-key xxx --output cloud-models.json
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

COMFY_CLOUD_BASE = "https://cloud.comfy.org"

MODEL_FOLDERS = ["unet", "clip", "vae", "checkpoints", "loras"]


def fetch_json(url: str, headers: dict = None, timeout: int = 60) -> object:
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def fetch_model_list(base: str, folder: str, api_key: str) -> list[str]:
    url = f"{base}/api/models/{folder}"
    try:
        result = fetch_json(url, headers={"X-API-Key": api_key})
        if isinstance(result, list):
            return sorted(result)
        return []
    except Exception as e:
        print(f"    WARNING: Could not fetch {folder}: {e}", file=sys.stderr)
        return []


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Comfy Cloud model file list")
    p.add_argument(
        "--api-key",
        default=os.environ.get("COMFY_CLOUD_API_KEY", ""),
        help="Comfy Cloud API key (or set COMFY_CLOUD_API_KEY env var)",
    )
    p.add_argument("--output", required=True, help="Output JSON file path")
    p.add_argument(
        "--cloud-base",
        default=os.environ.get("COMFY_CLOUD_ORG_BASE", COMFY_CLOUD_BASE),
    )
    args = p.parse_args()

    if not args.api_key:
        print(
            "ERROR: Comfy Cloud API key required.\n"
            "  Set COMFY_CLOUD_API_KEY or pass --api-key <key>",
            file=sys.stderr,
        )
        sys.exit(1)

    base = args.cloud_base.rstrip("/")
    models: dict[str, list[str]] = {}

    for folder in MODEL_FOLDERS:
        print(f"==> Fetching {folder} ...", file=sys.stderr)
        files = fetch_model_list(base, folder, args.api_key)
        models[folder] = files
        print(f"    {len(files)} files", file=sys.stderr)

    result = {
        "_comment": "Snapshot of model files available in Comfy Cloud. Do not edit by hand.",
        "_how_to_regenerate": "COMFY_CLOUD_API_KEY=xxx python comfy-cloud-setup/fetch_cloud_models.py --output comfy-cloud-setup/cloud-models.json",
        "comfy_cloud_base": base,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "models": models,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    total = sum(len(v) for v in models.values())
    print(f"\n==> Done. {total} total model files across {len(MODEL_FOLDERS)} folders.", file=sys.stderr)
    print(f"    Output written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
