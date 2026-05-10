#!/usr/bin/env bash
# =============================================================================
# setup.sh — ComfyUI installer/updater matching Comfy Cloud custom nodes
# =============================================================================
#
# TARGET: RTX 5090 / Blackwell (sm_120) + CUDA 13.0 on Windows (git bash)
#
# USAGE:
#   bash /path/to/comfy-cloud-setup/setup.sh --api-key YOUR_COMFY_CLOUD_KEY
#   bash /path/to/comfy-cloud-setup/setup.sh  # uses COMFY_CLOUD_API_KEY env var
#
# Run from the target ComfyUI install directory, e.g.:
#   mkdir -p /d/comfy-cloud && cd /d/comfy-cloud
#   bash /c/path/to/repo/comfy-cloud-setup/setup.sh --api-key xxx
#
# MODES:
#   Clean install — if main.py is NOT present in current dir:
#     Clones ComfyUI, creates venv, installs PyTorch + all cloud custom nodes.
#
#   Update — if main.py IS present in current dir:
#     Pulls latest ComfyUI, updates all custom nodes, refreshes requirements.
#
# SKIP FLAGS:
#   --skip-torch      Skip PyTorch reinstall (saves time on updates)
#   --skip-nodes      Skip custom node clone/pull (requirements still run)
#   --skip-triton     Skip triton override step
#   --python PATH     Python executable to use for venv (default: auto-detect)
#   --cuda VER        CUDA version for PyTorch index, e.g. 130 or 128 (default: 130)
#
# NOTES:
#   - Model files are NOT downloaded. Only code + pip packages.
#   - Qwen-TTS (firebirdxx/ComfyUI-Qwen-TTS) requires triton.  On Windows,
#     triton support is limited — the script tries triton-windows wheels as a
#     fallback.  If triton fails, Qwen-TTS nodes load but audio synthesis may
#     not work.  Comfy Cloud runs on Linux where triton works natively.
#   - ~10 internal Comfy Cloud nodes (radiance, test-framework, SchemaNodes …)
#     have no public repo and are skipped.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Script location (fetch_cloud_nodes.py lives alongside this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FETCH_SCRIPT="$SCRIPT_DIR/fetch_cloud_nodes.py"
FETCH_MODELS_SCRIPT="$SCRIPT_DIR/fetch_cloud_models.py"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
API_KEY="${COMFY_CLOUD_API_KEY:-}"
PYTHON_EXE=""          # auto-detect
CUDA_VER="130"         # cu130 = CUDA 13.0
SKIP_TORCH=false
SKIP_NODES=false
SKIP_TRITON=false

COMFY_REPO="https://github.com/comfyanonymous/ComfyUI.git"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)    API_KEY="$2";    shift 2 ;;
        --python)     PYTHON_EXE="$2"; shift 2 ;;
        --cuda)       CUDA_VER="$2";   shift 2 ;;
        --skip-torch)  SKIP_TORCH=true;  shift ;;
        --skip-nodes)  SKIP_NODES=true;  shift ;;
        --skip-triton) SKIP_TRITON=true; shift ;;
        -h|--help)
            sed -n '/^# ====/,/^# ===/p' "${BASH_SOURCE[0]}" | head -60
            exit 0
            ;;
        *) echo "Unknown option: $1  (use --help for usage)" >&2; exit 1 ;;
    esac
done

if [[ -z "$API_KEY" ]]; then
    echo ""
    echo "ERROR: Comfy Cloud API key is required."
    echo "  Pass --api-key YOUR_KEY  or  set COMFY_CLOUD_API_KEY in your environment."
    echo ""
    exit 1
fi

if [[ ! -f "$FETCH_SCRIPT" ]]; then
    echo "ERROR: fetch_cloud_nodes.py not found at $FETCH_SCRIPT"
    echo "  Keep setup.sh and fetch_cloud_nodes.py in the same directory."
    exit 1
fi

# ---------------------------------------------------------------------------
# Detect install mode
# ---------------------------------------------------------------------------
if [[ -f "main.py" && -d "comfy" ]]; then
    INSTALL_MODE="update"
    echo ""
    echo "=========================================="
    echo "  ComfyUI UPDATE mode"
    echo "  Directory: $(pwd)"
    echo "=========================================="
else
    INSTALL_MODE="clean"
    echo ""
    echo "=========================================="
    echo "  ComfyUI CLEAN INSTALL mode"
    echo "  Directory: $(pwd)"
    echo "=========================================="
fi

# ---------------------------------------------------------------------------
# Find Python
# ---------------------------------------------------------------------------
find_python() {
    # On Windows, `command -v python3` finds the Microsoft Store stub, but running
    # it exits with code 49.  Test each candidate by actually importing sys.
    local candidates=("${PYTHON_EXE:-}" py python3.12 python3.11 python3.10 python3 python)
    for py in "${candidates[@]}"; do
        [[ -z "$py" ]] && continue
        if "$py" -c "import sys" &>/dev/null 2>&1; then
            echo "$py"
            return 0
        fi
    done
    echo "ERROR: No working Python found. Install Python 3.12 first." >&2
    exit 1
}

PYTHON_BIN="$(find_python)"
PYTHON_VERSION="$("$PYTHON_BIN" --version 2>&1)"
echo ""
echo "==> Python: $PYTHON_BIN  ($PYTHON_VERSION)"

# Warn if Python < 3.10
PY_MINOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.minor)')"
PY_MAJOR="$("$PYTHON_BIN" -c 'import sys; print(sys.version_info.major)')"
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ) ]]; then
    echo "WARNING: Python 3.10+ recommended. Proceeding anyway."
fi

# ---------------------------------------------------------------------------
# Step 1: Clone or pull ComfyUI
# ---------------------------------------------------------------------------
echo ""
if [[ "$INSTALL_MODE" == "clean" ]]; then
    echo "==> Cloning ComfyUI into current directory..."
    # git clone requires directory to be empty.  Remove files that our own setup
    # process creates (log, json outputs) so they don't block the clone.
    for _f in setup.log cloud_nodes.json merged_requirements.txt; do
        [[ -f "$_f" ]] && { echo "  Removing pre-existing file: $_f"; rm -f "$_f"; }
    done
    if ! git clone "$COMFY_REPO" .; then
        echo ""
        echo "ERROR: git clone failed."
        echo "  The current directory must be empty (apart from setup.sh / fetch_cloud_nodes.py)."
        echo "  Create a new empty directory and run the script from there."
        exit 1
    fi
    echo "    ComfyUI cloned successfully."
else
    echo "==> Pulling latest ComfyUI..."
    git pull
fi

# ---------------------------------------------------------------------------
# Step 2: Set up Python virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="venv"
echo ""
if [[ -d "$VENV_DIR/Scripts" || -d "$VENV_DIR/bin" ]]; then
    echo "==> Venv already exists at ./$VENV_DIR"
else
    echo "==> Creating Python venv at ./$VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate (Windows git bash uses Scripts/activate, Linux uses bin/activate)
if [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/Scripts/activate"
elif [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Could not find venv activation script." >&2
    exit 1
fi

echo "    Active Python: $(python --version)  ($(which python))"

# ---------------------------------------------------------------------------
# Step 3: Upgrade pip + install build tools
# ---------------------------------------------------------------------------
echo ""
echo "==> Upgrading pip / wheel ..."
# setuptools intentionally not upgraded: PyTorch 2.11 requires setuptools<82
# and upgrading it triggers a noisy (but benign) pip resolver warning.
python -m pip install --upgrade pip wheel

# ---------------------------------------------------------------------------
# Step 4: Install PyTorch with CUDA support
# ---------------------------------------------------------------------------
if [[ "$SKIP_TORCH" == "true" ]]; then
    echo ""
    echo "==> Skipping PyTorch install (--skip-torch)."
else
    echo ""
    TORCH_INDEX="https://download.pytorch.org/whl/cu${CUDA_VER}"
    TORCH_INDEX_FALLBACK="https://download.pytorch.org/whl/cu128"
    echo "==> Installing PyTorch (CUDA ${CUDA_VER} / Blackwell RTX 5090) ..."
    echo "    Index: $TORCH_INDEX"

    # Try the requested CUDA version first, then fall back to cu128.
    if pip install torch torchvision torchaudio \
            --index-url "$TORCH_INDEX" 2>&1; then
        echo "    PyTorch installed with CUDA ${CUDA_VER}."
    else
        echo "    cu${CUDA_VER} index unavailable — trying CUDA 12.8 fallback ..."
        pip install torch torchvision torchaudio \
            --index-url "$TORCH_INDEX_FALLBACK"
        echo "    PyTorch installed with CUDA 12.8 (fallback)."
        echo "    NOTE: For full RTX 5090 (sm_120) support, check for a cu${CUDA_VER} build later."
    fi
fi

# ---------------------------------------------------------------------------
# Step 5: Install ComfyUI core requirements
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing ComfyUI core requirements ..."
pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Step 6: Fetch Comfy Cloud custom node list
# ---------------------------------------------------------------------------
# Use a file in the install dir so both bash and Python agree on the path
# (avoids /tmp vs Windows temp path mapping issues in git bash).
NODES_JSON="$(pwd)/cloud_nodes.json"

CLASSES_JSON="$(pwd)/cloud_node_classes.json"
# Repo path for the committed snapshot used by the workflow-audit skill and discovery agent.
REPO_CLASSES_JSON="$SCRIPT_DIR/cloud-node-classes.json"

echo ""
echo "==> Fetching Comfy Cloud node list (this contacts cloud.comfy.org) ..."
COMFY_CLOUD_API_KEY="$API_KEY" python "$FETCH_SCRIPT" \
    --output "$NODES_JSON" \
    --save-classes "$CLASSES_JSON"

# Copy class snapshot into the repo so audit tools and the discovery agent can use it
# without needing an API key or a live cloud connection.
if [[ -f "$CLASSES_JSON" ]]; then
    cp "$CLASSES_JSON" "$REPO_CLASSES_JSON"
    echo "    Repo snapshot updated: $REPO_CLASSES_JSON"
fi

# ---------------------------------------------------------------------------
# Step 6b: Fetch Comfy Cloud model file catalog
# ---------------------------------------------------------------------------
# Populates cloud-models.json alongside cloud-node-classes.json in the repo.
# Consumed by the workflow-audit skill (section 5b) to check model filenames.
REPO_MODELS_JSON="$SCRIPT_DIR/cloud-models.json"

echo ""
echo "==> Fetching Comfy Cloud model file list ..."
COMFY_CLOUD_API_KEY="$API_KEY" python "$FETCH_MODELS_SCRIPT" \
    --output "$REPO_MODELS_JSON"

# ---------------------------------------------------------------------------
# Step 7: Clone / update custom nodes
# ---------------------------------------------------------------------------
mkdir -p custom_nodes

if [[ "$SKIP_NODES" == "true" ]]; then
    echo ""
    echo "==> Skipping node git clone/pull (--skip-nodes)."
else
    echo ""
    echo "==> Installing/updating custom nodes ..."
    echo ""

    python - "$NODES_JSON" <<'PYEOF'
import json, shutil, subprocess, sys
from pathlib import Path

nodes_file = sys.argv[1]
with open(nodes_file, encoding="utf-8") as f:
    data = json.load(f)

custom_nodes_dir = Path("custom_nodes")
summary = data.get("summary", {})
nodes = data.get("nodes", [])
has_version_info = data.get("has_version_info", False)
nodes_with_versions = data.get("nodes_with_versions", 0)

installable = summary.get("matched", 0) + summary.get("manual_override", 0)
skipped = summary.get("internal_skip", 0) + summary.get("not_found", 0)
version_label = f"{nodes_with_versions} pinned to cloud commit" if has_version_info else "latest HEAD (version endpoint unavailable)"
print(f"  Cloud packages : {data.get('custom_node_packages', 0)}")
print(f"  To install     : {installable}  |  Skipping: {skipped}  |  Versions: {version_label}")
print()


def run_git(args, cwd=None, check=True):
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=True,
        text=True,
    )


cloud_node_names = set()
failed = []

for node in nodes:
    name = node["name"]
    git_url = node.get("git_url")
    pinned_hash = node.get("pinned_hash")
    status = node.get("status", "")

    if not git_url or status in ("internal_skip", "not_found"):
        label = "[SKIP-INTERNAL]" if status == "internal_skip" else "[NO-URL]      "
        print(f"  {label} {name}")
        continue

    cloud_node_names.add(name)
    node_dir = custom_nodes_dir / name
    hash_suffix = f"  -> {pinned_hash[:8]}" if pinned_hash else ""

    if node_dir.exists() and (node_dir / ".git").exists():
        print(f"  [UPDATE]  {name}{hash_suffix}")
        try:
            run_git(["fetch", "origin"], cwd=node_dir)
            if pinned_hash:
                run_git(["checkout", "--detach", pinned_hash], cwd=node_dir)
            else:
                # Reset hard to origin/HEAD — handles rebased branches and force-pushes
                # that silently break `git pull --ff-only`.
                ref_result = run_git(
                    ["symbolic-ref", "refs/remotes/origin/HEAD"],
                    cwd=node_dir, check=False,
                )
                default_branch = ref_result.stdout.strip().removeprefix("refs/remotes/") or "origin/main"
                run_git(["reset", "--hard", default_branch], cwd=node_dir)
            run_git(["submodule", "update", "--init", "--recursive"], cwd=node_dir, check=False)
        except subprocess.CalledProcessError as e:
            print(f"    WARNING: Update failed for {name} — {e.stderr.strip()[:200]}")
            failed.append(name)

    else:
        # Remove non-git directory leftovers so clone can proceed
        if node_dir.exists() and not (node_dir / ".git").exists():
            print(f"    Removing non-git directory: {node_dir}")
            shutil.rmtree(node_dir)

        print(f"  [INSTALL] {name}{hash_suffix}")
        print(f"            {git_url}")
        try:
            if pinned_hash:
                # Need full history to checkout a specific past commit
                run_git(["clone", "--recurse-submodules", git_url, str(node_dir)])
                run_git(["checkout", "--detach", pinned_hash], cwd=node_dir)
            else:
                run_git(["clone", "--depth=1", "--recurse-submodules", git_url, str(node_dir)])
        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Clone failed for {name} — {e.stderr.strip()[:200]}")
            failed.append(name)

# Stale node detection — warn about local git dirs absent from the current cloud list.
# Does not auto-delete; removes are intentional and irreversible.
ALWAYS_LOCAL = {"ComfyUI-Manager", ".cache"}
stale = [
    d.name for d in custom_nodes_dir.iterdir()
    if d.is_dir() and (d / ".git").exists()
    and d.name not in cloud_node_names
    and d.name not in ALWAYS_LOCAL
] if custom_nodes_dir.exists() else []

print()
print("  -- Summary --------------------------------------------------")
if failed:
    print(f"  FAILED  : {len(failed)} node(s) - {failed}")
if stale:
    print(f"  STALE   : {len(stale)} local node(s) not in current cloud install:")
    for s in stale:
        print(f"              custom_nodes/{s}")
    print("            (to remove: rm -rf custom_nodes/<name>  then re-run)")
ok_count = len(cloud_node_names) - len(failed)
print(f"  OK      : {ok_count} node(s) installed/updated")
print("  -------------------------------------------------------------")
PYEOF
fi

# ---------------------------------------------------------------------------
# Step 7b: Patch ComfyUI-Qwen-TTS — replace PGCRT nodes with FB_ cloud nodes
# ---------------------------------------------------------------------------
# Comfy Cloud runs a private fork with FB_-prefixed class names (FB_Qwen3TTSVoiceDesign
# etc.).  We cloned PGCRT's public equivalent above as a base, but its nodes use
# different class names and a model-socket architecture that doesn't match cloud
# workflows.  We overwrite __init__.py and inject fb_cloud_nodes.py so that only
# the FB_ nodes (identical schema to cloud) are registered with ComfyUI.
QWEN_DIR="custom_nodes/ComfyUI-Qwen-TTS"
FB_NODES_SRC="$SCRIPT_DIR/fb_cloud_nodes.py"

if [[ -d "$QWEN_DIR" ]]; then
    echo ""
    echo "==> Patching ComfyUI-Qwen-TTS with FB_ cloud-compatible nodes ..."

    if [[ ! -f "$FB_NODES_SRC" ]]; then
        echo "  ERROR: fb_cloud_nodes.py not found at $FB_NODES_SRC"
        echo "  Keep fb_cloud_nodes.py in the same directory as setup.sh."
        exit 1
    fi

    cp "$FB_NODES_SRC" "$QWEN_DIR/fb_cloud_nodes.py"
    echo "  Copied fb_cloud_nodes.py"

    # Download flybirdxx nodes.py to get SaveVoiceNode / LoadSpeakerNode.
    # These are not in PGCRT's fork and are suppressed by default — we add them
    # alongside the FB_ cloud nodes so voice save/load works locally.
    echo "  Downloading nodes.py from flybirdxx/ComfyUI-Qwen-TTS ..."
    curl -sSL \
        "https://raw.githubusercontent.com/flybirdxx/ComfyUI-Qwen-TTS/main/nodes.py" \
        -o "$QWEN_DIR/nodes.py" \
        && echo "  Downloaded nodes.py" \
        || echo "  WARNING: Could not download nodes.py — SaveVoiceNode/LoadSpeakerNode will be unavailable."

    cat > "$QWEN_DIR/__init__.py" <<'INITEOF'
"""
ComfyUI Qwen3-TTS Nodes — Comfy Cloud FB_ compatibility layer + standard nodes.

FB_-prefixed nodes match Comfy Cloud workflow schema exactly.
SaveVoiceNode / LoadSpeakerNode added for voice persistence (save/load .qvp files).
"""

from .fb_cloud_nodes import FB_NODE_CLASS_MAPPINGS, FB_NODE_DISPLAY_NAME_MAPPINGS
from .nodes import SaveVoiceNode, LoadSpeakerNode

NODE_CLASS_MAPPINGS = {
    **FB_NODE_CLASS_MAPPINGS,
    "SaveVoiceNode": SaveVoiceNode,
    "LoadSpeakerNode": LoadSpeakerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **FB_NODE_DISPLAY_NAME_MAPPINGS,
    "SaveVoiceNode": "Qwen3-TTS Save Voice [New]",
    "LoadSpeakerNode": "Qwen3-TTS Load Speaker [New]",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
INITEOF
    echo "  Wrote __init__.py (FB_ + SaveVoice/LoadSpeaker)"
else
    echo ""
    echo "  WARNING: $QWEN_DIR not found — Qwen-TTS patch skipped."
fi

# ---------------------------------------------------------------------------
# Step 8: Install all custom node requirements
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing custom node pip requirements ..."
echo "    (collecting all requirements.txt files, then resolving together)"
echo ""

# Collect all requirements into one merged file for joint resolution.
MERGED_REQS="$(pwd)/merged_requirements.txt"

# Write a header
echo "# Auto-merged requirements from ComfyUI custom nodes" > "$MERGED_REQS"
echo "# Generated by comfy-cloud-setup/setup.sh" >> "$MERGED_REQS"

REQS_FOUND=0
for req_file in custom_nodes/*/requirements.txt; do
    [[ -f "$req_file" ]] || continue
    node_dir="$(basename "$(dirname "$req_file")")"
    echo "" >> "$MERGED_REQS"
    echo "# --- $node_dir ---" >> "$MERGED_REQS"
    # Strip blank lines and comments to keep the file clean
    grep -v '^\s*#' "$req_file" | grep -v '^\s*$' >> "$MERGED_REQS" || true
    REQS_FOUND=$((REQS_FOUND + 1))
done

echo "==> Found requirements.txt in $REQS_FOUND custom node(s)."
echo "    Running joint pip resolve (pip will negotiate version conflicts) ..."
echo ""

# Run joint resolve — don't abort on conflict, just report
pip install -r "$MERGED_REQS" 2>&1 || {
    echo ""
    echo "WARNING: pip reported errors during joint resolve."
    echo "  This is usually a version conflict between custom nodes."
    echo "  Trying individual installs as fallback ..."
    echo ""
    for req_file in custom_nodes/*/requirements.txt; do
        [[ -f "$req_file" ]] || continue
        node_dir="$(basename "$(dirname "$req_file")")"
        echo "  Installing: $node_dir"
        pip install -r "$req_file" 2>&1 | tail -3 || true
    done
}

# ---------------------------------------------------------------------------
# Step 8b: Known missing packages (caught from first-run error analysis)
# ---------------------------------------------------------------------------
echo ""
echo "==> Installing known-missing packages not declared in requirements.txt ..."

# torch_complex: required by ComfyUI_AudioTools (look2hear audio separation)
pip install torch-complex --quiet

# ml_dtypes>=0.5.0: onnx requires float4_e2m1fn which was added in 0.5.0
# Upgrading onnx alongside to ensure compatibility.
pip install --upgrade "ml_dtypes>=0.5.0" onnx --quiet

# onnxruntime-gpu: replaces onnxruntime with CUDA execution providers
# (fixes DWPose slow-CPU warning in comfyui_controlnet_aux)
pip install onnxruntime-gpu --quiet 2>/dev/null || \
    echo "    onnxruntime-gpu not available for this CUDA version — DWPose will use CPU."

# sageattention: faster attention for TTS and other nodes (optional, graceful fallback to sdpa)
pip install sageattention --quiet 2>/dev/null || \
    echo "    sageattention not available — sdpa will be used instead."

echo "    Done."

# ---------------------------------------------------------------------------
# Step 8c: Qwen-TTS — install without deps, then restore transformers
# ---------------------------------------------------------------------------
# qwen-tts pins transformers==4.57.3, but that conflicts with other nodes that
# need transformers 5.x.  Also, transformers 5.x changed check_model_inputs from
# a decorator factory (@check_model_inputs()) to a plain decorator, causing a
# TypeError at import time if qwen-tts's own pip deps are resolved.
# Fix: install qwen-tts without its deps, then pin transformers to a version that
# satisfies both qwen-tts and other nodes (4.51+ has the factory API; 4.57.x is
# the last 4.x release).
echo ""
echo "==> Installing qwen-tts (transformers-compatible) ..."
# Install without deps to avoid the transformers==4.57.3 pin downgrading 5.x.
pip install qwen-tts --no-deps --quiet
# qwen-tts needs transformers with check_model_inputs as a decorator factory.
# 4.x has the factory API; 5.x changed it. Pin to latest 4.x.
pip install "transformers>=4.50,<5.0" "huggingface-hub>=0.24" --quiet
echo "    qwen-tts installed, transformers pinned to 4.x."

# ---------------------------------------------------------------------------
# Step 9: Triton — CUDA 13.0 / Blackwell override
# ---------------------------------------------------------------------------
if [[ "$SKIP_TRITON" == "true" ]]; then
    echo ""
    echo "==> Skipping triton override (--skip-triton)."
else
    echo ""
    echo "==> Triton install for CUDA ${CUDA_VER} / Blackwell ..."
    echo "    Qwen-TTS requires triton. On Linux (like Comfy Cloud) this works natively."
    echo "    On Windows, triton support is limited — trying multiple approaches."
    echo ""

    TRITON_OK=false

    # Approach 1: standard triton from PyPI (works on Linux, may fail on Windows)
    if pip install "triton>=5.0" --quiet 2>/dev/null; then
        echo "    triton>=5.0 installed from PyPI."
        TRITON_OK=true
    fi

    # Approach 2: triton-windows wheels (community Windows builds)
    if [[ "$TRITON_OK" == "false" ]]; then
        echo "    Standard triton failed. Trying triton-windows ..."
        if pip install triton-windows --quiet 2>/dev/null; then
            echo "    triton-windows installed."
            TRITON_OK=true
        fi
    fi

    # Approach 3: nightly/prebuilt wheels for Blackwell
    if [[ "$TRITON_OK" == "false" ]]; then
        echo "    triton-windows failed. Trying PyTorch nightly triton wheel ..."
        if pip install triton --index-url https://download.pytorch.org/whl/nightly/cu${CUDA_VER} --quiet 2>/dev/null; then
            echo "    triton from PyTorch nightly installed."
            TRITON_OK=true
        fi
    fi

    if [[ "$TRITON_OK" == "false" ]]; then
        echo ""
        echo "    WARNING: Could not install triton on Windows."
        echo "    Qwen-TTS synthesis will likely fail to initialize on this platform."
        echo "    All other custom nodes should work normally."
        echo ""
        echo "    Options:"
        echo "      a) Use WSL2 (Linux) where triton works natively."
        echo "      b) Use Comfy Cloud for Qwen-TTS workflows."
        echo "      c) Check https://github.com/woct0rdho/triton-windows for"
        echo "         community Blackwell Windows wheels when they become available."
    fi
fi

# ---------------------------------------------------------------------------
# Step 10: Install ComfyUI Manager (optional but useful for local management)
# ---------------------------------------------------------------------------
MANAGER_DIR="custom_nodes/ComfyUI-Manager"
echo ""
if [[ -d "$MANAGER_DIR/.git" ]]; then
    echo "==> Updating ComfyUI-Manager ..."
    git -C "$MANAGER_DIR" pull --ff-only 2>/dev/null || echo "    (already up to date)"
elif [[ ! -d "$MANAGER_DIR" ]]; then
    echo "==> Installing ComfyUI-Manager (local node management) ..."
    git clone --depth=1 \
        "https://github.com/ltdrdata/ComfyUI-Manager.git" \
        "$MANAGER_DIR" 2>&1 | tail -2 || \
        echo "    WARNING: ComfyUI-Manager clone failed (non-critical)."
fi

# ---------------------------------------------------------------------------
# Step 11: Verification
# ---------------------------------------------------------------------------
echo ""
echo "==> Verifying PyTorch + CUDA ..."
python - <<'PYEOF'
import sys
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"    PyTorch  : {torch.__version__}")
    print(f"    CUDA avail: {cuda_ok}")
    if cuda_ok:
        print(f"    CUDA ver : {torch.version.cuda}")
        print(f"    GPU      : {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"    Compute  : sm_{cap[0]}{cap[1]}")
        if cap[0] < 12:
            print("    WARNING: sm_12x (Blackwell) not detected.")
    else:
        print("    WARNING: CUDA not available — GPU won't be used.")
except ImportError:
    print("    ERROR: PyTorch not importable. Check install.")
    sys.exit(1)

try:
    import triton
    print(f"    Triton   : {triton.__version__}")
except ImportError:
    print("    Triton   : not installed (Qwen-TTS won't work)")
PYEOF

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "  To run ComfyUI:"
if [[ -f "venv/Scripts/activate" ]]; then
    echo "    cd \"$(pwd)\""
    echo "    source venv/Scripts/activate"
else
    echo "    cd \"$(pwd)\""
    echo "    source venv/bin/activate"
fi
echo "    python main.py --listen"
echo ""
echo "  Notes:"
echo "  - No model files were downloaded. Copy/symlink your models/ directory."
echo "  - Run this script again to update ComfyUI + all custom nodes."
echo "  - Cloud-only nodes skipped (radiance, SchemaNodes, test-framework, etc.)"
echo "    These exist only on Comfy Cloud's proprietary installation."
echo ""
