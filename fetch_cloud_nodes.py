#!/usr/bin/env python3
"""
fetch_cloud_nodes.py — Query Comfy Cloud object_info and resolve GitHub URLs.

Fetches the node list from Comfy Cloud, extracts which custom node packages are
installed (via the python_module field), then cross-references the ComfyUI Manager
public registry to find the GitHub clone URL for each.

Also attempts to fetch installed node versions from the Comfy Cloud Manager endpoint
so the installer can pin to the exact same commit the cloud is running. This is
best-effort — Comfy Cloud may not proxy the Manager endpoint.

Outputs a JSON file consumed by setup.sh.

Usage:
    COMFY_CLOUD_API_KEY=xxx python fetch_cloud_nodes.py --output nodes.json
    python fetch_cloud_nodes.py --api-key xxx --output nodes.json
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMFY_CLOUD_BASE = "https://cloud.comfy.org"
OBJECT_INFO_PATH = "/api/object_info"

MANAGER_NODE_LIST_URL = (
    "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/custom-node-list.json"
)

# Manual overrides: cloud directory name -> correct GitHub URL (or None to skip).
# Covers partial-match errors, nodes not in Manager registry, and internal nodes.
MANUAL_OVERRIDES: dict[str, Optional[str]] = {
    # Qwen-TTS: Comfy Cloud runs a private fork with FB_-prefixed class names.
    # The original author (firebirdxx) deleted their account. PGCRT's repo is the
    # closest public equivalent — same nodes, minus the FB_ prefix. setup.sh patches
    # __init__.py afterwards to restore FB_ compatibility.
    "ComfyUI-Qwen-TTS": "https://github.com/PGCRT/ComfyUI-QWEN3_TTS",
    # Logic nodes: Manager matched ComfyUI-LogicUtils but cloud dir is comfyui-logic.
    "comfyui-logic": "https://github.com/aria1th/ComfyUI-LogicUtils",
    # NVIDIA RTX nodes: repo URL unconfirmed — skipping until correct URL is found.
    "comfyui_nvidia_rtx_nodes": None,
    # UltraShape: repo not found at expected URL — skipping until correct URL is found.
    "ComfyUI-UltraShape1": None,
    # Internal Comfy Cloud nodes — no public repo, skip.
    "radiance": None,
    "ComfyUI-EditUtils": None,
    "ComfyUI-test-framework": None,
    "ComfyUI_SchemaNodes": None,
    "websocket_image_save": None,   # Built into ComfyUI core.
    "vewd": None,
    "comfyui-systms-facecomposite": None,
    "feedback-sampler": None,
    # ComfyUI-WanAnimatePreprocess: not in Manager registry and correct repo unknown.
    # Skipping to avoid cloning WanVideoWrapper twice (which causes a duplicate warning).
    "ComfyUI-WanAnimatePreprocess": None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def fetch_json(url: str, headers: dict = None, timeout: int = 120) -> object:
    req = urllib.request.Request(url, headers=headers or {})
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def normalize(name: str) -> str:
    """Lowercase, strip all separators — used for fuzzy name matching."""
    return re.sub(r"[-_. ]", "", name.lower())


def extract_custom_node_dirs(object_info: dict) -> list[str]:
    """Return custom node directory names (sorted by node count desc)."""
    dirs: Counter = Counter()
    for info in object_info.values():
        if not isinstance(info, dict):
            continue
        mod = info.get("python_module", "")
        if isinstance(mod, str) and mod.startswith("custom_nodes."):
            parts = mod.split(".")
            if len(parts) >= 2:
                dirs[parts[1]] += 1
    return [name for name, _ in sorted(dirs.items(), key=lambda x: -x[1])]


def build_manager_index(manager_list: list) -> dict:
    """
    Map normalize(repo_name) -> {git_url, title, repo_name}.
    Uses the last path component of the reference/files URL as repo_name.
    """
    index: dict = {}
    for entry in manager_list:
        if not isinstance(entry, dict):
            continue
        ref = entry.get("reference", "")
        if not ref:
            files = entry.get("files", [])
            ref = files[0] if files else ""
        if not ref:
            continue
        repo_name = ref.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        key = normalize(repo_name)
        if key not in index:
            index[key] = {
                "git_url": ref,
                "title": entry.get("title", repo_name),
                "repo_name": repo_name,
            }
    return index


def find_git_url(dir_name: str, index: dict) -> Optional[str]:
    """Exact match then substring fallback. Returns None if no match."""
    key = normalize(dir_name)
    if key in index:
        return index[key]["git_url"]
    # Substring fallback — requires key length > 6 to avoid noise.
    if len(key) > 6:
        for k, v in index.items():
            if key in k or k in key:
                return v["git_url"]
    return None


def is_hex_hash(value: str) -> bool:
    """Return True if value looks like a git commit hash (7–40 hex chars)."""
    return bool(value) and 7 <= len(value) <= 40 and all(c in "0123456789abcdefABCDEF" for c in value)


def try_fetch_node_versions(base: str, api_key: str) -> dict[str, str]:
    """
    Attempt to fetch installed node git hashes from the Comfy Cloud Manager endpoint.

    Comfy Cloud may or may not proxy the ComfyUI Manager /customnode/getlist endpoint.
    Returns {normalize(repo_name): commit_hash} for any nodes where a hash is found.
    Returns empty dict if the endpoint is unavailable or returns no version data.
    """
    candidate_urls = [
        f"{base}/customnode/getlist?mode=local",
        f"{base}/api/customnode/getlist?mode=local",
    ]
    headers = {"X-API-Key": api_key}

    for url in candidate_urls:
        try:
            data = fetch_json(url, headers=headers, timeout=20)
        except Exception:
            continue

        nodes = data.get("custom_nodes") if isinstance(data, dict) else None
        if not isinstance(nodes, list) or not nodes:
            continue

        result: dict[str, str] = {}
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("installed") not in ("True", True):
                continue

            # Try direct hash fields first
            git_hash = None
            for field in ("hash", "git_hash", "version"):
                val = node.get(field, "")
                if isinstance(val, str) and is_hex_hash(val):
                    git_hash = val
                    break

            # Fall back to cnr_info (ComfyUI Node Registry installed version)
            if not git_hash:
                cnr = node.get("cnr_info", {})
                if isinstance(cnr, dict):
                    iv = cnr.get("installed_version", {})
                    if isinstance(iv, dict):
                        for field in ("version", "hash"):
                            val = iv.get(field, "")
                            if isinstance(val, str) and is_hex_hash(val):
                                git_hash = val
                                break

            if not git_hash:
                continue

            # Key by normalized repo name from the reference URL
            ref = node.get("reference", "") or (node.get("files") or [""])[0]
            repo_name = ref.rstrip("/").split("/")[-1].removesuffix(".git")
            if repo_name:
                result[normalize(repo_name)] = git_hash

        if result:
            return result

    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch Comfy Cloud custom node list")
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
    p.add_argument(
        "--no-manager",
        action="store_true",
        help="Skip ComfyUI Manager registry lookup (use overrides only)",
    )
    p.add_argument(
        "--no-versions",
        action="store_true",
        help="Skip version/hash fetch attempt (always use latest HEAD)",
    )
    p.add_argument(
        "--save-classes",
        metavar="PATH",
        default=None,
        help="Also save a sorted list of ALL node class names from object_info to this path",
    )
    args = p.parse_args()

    if not args.api_key:
        print(
            "ERROR: Comfy Cloud API key required.\n"
            "  Set COMFY_CLOUD_API_KEY or pass --api-key <key>",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Fetch object_info from Comfy Cloud
    print("==> Fetching Comfy Cloud object_info ...", file=sys.stderr)
    base = args.cloud_base.rstrip("/")
    try:
        object_info = fetch_json(
            f"{base}{OBJECT_INFO_PATH}",
            headers={"X-API-Key": args.api_key},
            timeout=120,
        )
    except Exception as e:
        print(f"ERROR: Could not fetch Comfy Cloud object_info: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(object_info, dict):
        print("ERROR: Unexpected response format from Comfy Cloud.", file=sys.stderr)
        sys.exit(1)

    print(f"    {len(object_info)} total nodes in Comfy Cloud.", file=sys.stderr)

    custom_node_dirs = extract_custom_node_dirs(object_info)
    print(f"    {len(custom_node_dirs)} custom node packages detected.", file=sys.stderr)

    # 2. Fetch ComfyUI Manager registry (for GitHub URLs)
    manager_index: dict = {}
    if not args.no_manager:
        print("==> Fetching ComfyUI Manager registry ...", file=sys.stderr)
        try:
            raw = fetch_json(MANAGER_NODE_LIST_URL, timeout=60)
            manager_list = (
                raw.get("custom_nodes", raw) if isinstance(raw, dict) else raw
            )
            if not isinstance(manager_list, list):
                manager_list = []
            manager_index = build_manager_index(manager_list)
            print(f"    {len(manager_index)} entries in Manager registry.", file=sys.stderr)
        except Exception as e:
            print(f"    WARNING: Could not fetch Manager registry: {e}", file=sys.stderr)

    # 3. Attempt to fetch installed node versions (git hashes) from Comfy Cloud
    node_versions: dict[str, str] = {}
    if not args.no_versions:
        print("==> Fetching Comfy Cloud node versions (Manager endpoint, best-effort) ...", file=sys.stderr)
        node_versions = try_fetch_node_versions(base, args.api_key)
        if node_versions:
            print(f"    Got commit hashes for {len(node_versions)} node(s) — installer will pin to these.", file=sys.stderr)
        else:
            print("    Version endpoint unavailable — installer will use latest HEAD of each node.", file=sys.stderr)

    # 4. Match each custom node dir to a GitHub URL and optional pinned hash
    nodes = []
    not_found = []
    skipped_internal = []

    for dir_name in custom_node_dirs:
        # Look up pinned hash by both the cloud dir name and any partial matches
        pinned = node_versions.get(normalize(dir_name))

        # Manual overrides take precedence for URL resolution
        if dir_name in MANUAL_OVERRIDES:
            override = MANUAL_OVERRIDES[dir_name]
            if override is None:
                skipped_internal.append(dir_name)
                nodes.append(
                    {"name": dir_name, "git_url": None, "status": "internal_skip", "pinned_hash": None}
                )
            else:
                nodes.append(
                    {"name": dir_name, "git_url": override, "status": "manual_override", "pinned_hash": pinned}
                )
            continue

        git_url = find_git_url(dir_name, manager_index)
        if git_url:
            nodes.append({"name": dir_name, "git_url": git_url, "status": "matched", "pinned_hash": pinned})
        else:
            not_found.append(dir_name)
            nodes.append({"name": dir_name, "git_url": None, "status": "not_found", "pinned_hash": None})

    # 5a. Optionally save full node class list
    if args.save_classes:
        all_classes = sorted(k for k in object_info.keys() if isinstance(k, str))
        classes_out = {
            "comfy_cloud_base": base,
            "total_node_classes": len(all_classes),
            "node_classes": all_classes,
        }
        with open(args.save_classes, "w", encoding="utf-8") as f:
            json.dump(classes_out, f, indent=2)
        print(f"    Node class list ({len(all_classes)} entries) saved to: {args.save_classes}", file=sys.stderr)

    # 5b. Write package list output
    nodes_with_versions = sum(1 for n in nodes if n.get("pinned_hash"))
    result = {
        "comfy_cloud_base": base,
        "total_nodes_in_cloud": len(object_info),
        "custom_node_packages": len(custom_node_dirs),
        "has_version_info": bool(node_versions),
        "nodes_with_versions": nodes_with_versions,
        "nodes": nodes,
        "summary": {
            "matched": sum(1 for n in nodes if n["status"] == "matched"),
            "manual_override": sum(1 for n in nodes if n["status"] == "manual_override"),
            "internal_skip": len(skipped_internal),
            "not_found": len(not_found),
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # 6. Print summary
    s = result["summary"]
    print(f"\n==> Node resolution summary:", file=sys.stderr)
    print(f"    Matched via registry : {s['matched']}", file=sys.stderr)
    print(f"    Matched via override : {s['manual_override']}", file=sys.stderr)
    print(f"    Skipped (internal)   : {s['internal_skip']} — {skipped_internal}", file=sys.stderr)
    if not_found:
        print(f"    Not found            : {len(not_found)} — {not_found}", file=sys.stderr)
    if node_versions:
        print(f"    Version-pinned       : {nodes_with_versions} node(s)", file=sys.stderr)
    print(f"\n    Output written to: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
