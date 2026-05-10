# comfyui-cloud-parity

Install and update a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) environment that mirrors the custom nodes available on [Comfy Cloud](https://comfy.icu). Run the same workflows locally that you've developed (or tested) on the cloud, without manually tracking which nodes are installed there.

## Stability note

This project uses **unofficial, undocumented Comfy Cloud APIs** to discover which custom nodes and models are currently installed on the cloud. These APIs are not part of any public contract and may change without notice. If the script stops working or produces errors, the most likely cause is an API change.

## What it does

- **Clean install** — clones ComfyUI, creates a Python venv, installs PyTorch with CUDA support, then installs every custom node currently on Comfy Cloud
- **Update** — pulls the latest ComfyUI and refreshes all custom nodes to match the current cloud state
- Queries the Comfy Cloud API to get the exact node list at run time, so the install stays in sync as the cloud evolves
- Does **not** download model files — only code and pip packages

Tested on Windows with Git Bash and an NVIDIA RTX GPU (CUDA 12.8 / 13.0). Linux should work with minor path adjustments.

## Prerequisites

- Windows (Git Bash) or Linux/macOS
- Python 3.10+
- Git
- NVIDIA GPU with CUDA drivers installed
- A [Comfy Cloud](https://comfy.icu) account — the API key is used to fetch the current node list

## Quick start

```bash
# 1. Clone this repo somewhere accessible
git clone https://github.com/alankent/comfyui-cloud-parity.git ~/git/comfyui-cloud-parity

# 2. Create a directory for your ComfyUI installation and run the setup
mkdir -p ~/comfy-local && cd ~/comfy-local
bash ~/git/comfyui-cloud-parity/setup.sh --api-key YOUR_COMFY_CLOUD_KEY

# 3. Start ComfyUI
bash ~/git/comfyui-cloud-parity/start.sh
```

ComfyUI will be available at http://localhost:8188.

## Updating

Pull the latest setup script, then re-run from your ComfyUI directory:

```bash
cd ~/git/comfyui-cloud-parity && git pull
cd ~/comfy-local
bash ~/git/comfyui-cloud-parity/setup.sh --api-key YOUR_COMFY_CLOUD_KEY --skip-torch
```

`--skip-torch` skips the PyTorch reinstall to save time on updates.

## Models

Model files are **not** downloaded by this script — only code. Copy or symlink your `models/` directory from an existing ComfyUI install, or download models manually from [HuggingFace](https://huggingface.co) or [Civitai](https://civitai.com).

## Why does it need a Comfy Cloud API key?

The script queries the Comfy Cloud API to get the exact list of custom nodes currently installed on the cloud, then mirrors that list locally. This is what keeps your local install in sync as the cloud adds or removes nodes.

## Options

```
--api-key KEY     Comfy Cloud API key (or set COMFY_CLOUD_API_KEY env var)
--skip-torch      Skip PyTorch reinstall (faster updates)
--skip-nodes      Skip custom node clone/pull
--skip-triton     Skip triton install
--cuda VER        CUDA version, e.g. 128 or 130 (default: 130)
--python PATH     Python executable to use
```

## Notes

- A small number of Comfy Cloud nodes are internal/proprietary and have no public repo — these are skipped with a warning
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) is installed automatically for local node management
- Qwen3-TTS nodes are patched to use the same `FB_`-prefixed class names as Comfy Cloud, so TTS workflows run identically on cloud and local

## License

Apache 2.0 — see [LICENSE](LICENSE).
