"""
fb_cloud_nodes.py — Comfy Cloud FB_-prefix compatibility layer for Qwen3-TTS.

Comfy Cloud runs a private fork of the Qwen3-TTS node with all class names
prefixed with "FB_" and a self-contained architecture (no separate model-loader
node; each generation node loads the model internally via model_choice).

This file implements all 6 FB_ nodes with exact input/output schemas matching
Comfy Cloud's object_info, so workflows saved from Comfy Cloud run unchanged.

Node mapping:
    FB_Qwen3TTSVoiceDesign        — text + style instruction → AUDIO
    FB_Qwen3TTSCustomVoice        — text + predefined speaker → AUDIO
    FB_Qwen3TTSVoiceClone         — text + ref audio → AUDIO
    FB_Qwen3TTSVoiceClonePrompt   — ref audio → VOICE_CLONE_PROMPT (reusable)
    FB_Qwen3TTSRoleBank           — up to 8 (name, prompt) pairs → QWEN3_ROLE_BANK
    FB_Qwen3TTSDialogueInference  — script + role bank → AUDIO (multi-character)
"""

from __future__ import annotations

import gc
import re
import time

import numpy as np
import torch

try:
    import folder_paths
    _HAS_FOLDER_PATHS = True
except ImportError:
    _HAS_FOLDER_PATHS = False

try:
    from qwen_tts import Qwen3TTSModel
    _QWEN_TTS_OK = True
except Exception as _qwen_err:
    _QWEN_TTS_OK = False
    print(f"[FB_Qwen3TTS] WARNING: qwen-tts unavailable ({type(_qwen_err).__name__}: {_qwen_err})")
    print("[FB_Qwen3TTS] Nodes will register but generation will fail until fixed.")

try:
    from huggingface_hub import snapshot_download
    _HF_HUB_OK = True
except ImportError:
    _HF_HUB_OK = False

# ---------------------------------------------------------------------------
# Model ID resolution
# ---------------------------------------------------------------------------

_MODEL_IDS = {
    # (size, mode) -> HuggingFace repo id
    ("0.6B", "custom"):  "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    ("0.6B", "design"):  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",  # no 0.6B design
    ("0.6B", "clone"):   "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    ("1.7B", "custom"):  "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ("1.7B", "design"):  "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ("1.7B", "clone"):   "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

_ATTN_MAP = {
    "auto":       "sdpa",             # sdpa is built into PyTorch 2.x, always available
    "sage_attn":  "sdpa",             # resolved dynamically in _resolve_attn
    "flash_attn": "flash_attention_2",
    "sdpa":       "sdpa",
    "eager":      "eager",
}

# Per-process model cache: (hf_model_id, device, dtype_str) -> Qwen3TTSModel
_MODEL_CACHE: dict = {}


def _resolve_attn(attention: str) -> str:
    mapped = _ATTN_MAP.get(attention, "sdpa")
    if attention == "sage_attn":
        try:
            import sageattention  # noqa: F401
            mapped = "eager"  # sageattention patches ops globally; use eager mode with it
        except ImportError:
            mapped = "sdpa"
    if mapped == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("[FB_Qwen3TTS] flash-attn not installed, falling back to sdpa")
            mapped = "sdpa"
    return mapped


def _model_dir() -> str:
    """Return path to TTS model storage directory."""
    if _HAS_FOLDER_PATHS:
        import os
        return os.path.join(folder_paths.models_dir, "TTS")
    import os
    return os.path.join(os.getcwd(), "models", "TTS")


def _load_model(model_choice: str, mode: str, device: str, precision: str,
                attention: str, unload_after: bool = False) -> "Qwen3TTSModel":
    """Load (or retrieve cached) Qwen3TTSModel."""
    if not _QWEN_TTS_OK:
        raise RuntimeError("qwen-tts package not installed. Run: pip install qwen-tts")

    import os
    hf_id = _MODEL_IDS[(model_choice, mode)]
    model_folder = hf_id.split("/")[-1]
    local_path = os.path.join(_model_dir(), model_folder)

    # Download if not present
    if not os.path.isdir(local_path) or not os.listdir(local_path):
        if _HF_HUB_OK:
            print(f"[FB_Qwen3TTS] Downloading {hf_id} → {local_path}")
            snapshot_download(repo_id=hf_id, local_dir=local_path)
        else:
            print(f"[FB_Qwen3TTS] huggingface_hub not available; will attempt to load {hf_id} from HF cache")
            local_path = hf_id

    load_path = local_path if (os.path.isdir(local_path) and os.listdir(local_path)) else hf_id
    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    dtype = dtype_map.get(precision, torch.bfloat16)
    attn_impl = _resolve_attn(attention)

    cache_key = (load_path, device, str(dtype), attn_impl)

    if unload_after and cache_key in _MODEL_CACHE:
        del _MODEL_CACHE[cache_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    print(f"[FB_Qwen3TTS] Loading {load_path} (device={device}, dtype={precision}, attn={attn_impl})")
    model = Qwen3TTSModel.from_pretrained(
        load_path,
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_impl,
    )
    _MODEL_CACHE[cache_key] = model
    return model


def _set_seed(seed: int) -> None:
    if seed == 0:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _audio_from_wavs(wavs, sr: int) -> dict:
    waveform = torch.from_numpy(wavs[0]).unsqueeze(0).unsqueeze(0)
    return {"waveform": waveform, "sample_rate": sr}


def _audio_array(ref_audio: dict) -> tuple[np.ndarray, int]:
    waveform = ref_audio["waveform"].squeeze(0).cpu().numpy().copy()
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
    return waveform.astype(np.float32), ref_audio["sample_rate"]


def _silence_samples(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# FB_ node implementations
# ---------------------------------------------------------------------------

class FB_Qwen3TTSVoiceDesign:
    """Qwen3-TTS: generate audio from text + style instruction (voice design)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":         ("STRING", {"multiline": True, "default": "Hello world",
                                            "placeholder": "Enter text to synthesize"}),
                "instruct":     ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Style instruction (required for VoiceDesign)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device":       (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision":    (["bf16", "fp32"], {"default": "bf16"}),
                "language":     (["Auto", "Chinese", "English", "Japanese", "Korean",
                                  "French", "German", "Spanish", "Portuguese", "Russian", "Italian"],
                                 {"default": "Auto"}),
            },
            "optional": {
                "seed":          ("INT", {"default": 0, "min": 0, "max": 2**64 - 1,
                                          "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p":          ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k":          ("INT",   {"default": 20, "min": 0, "max": 100}),
                "temperature":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
                "attention":      (["auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                                   {"default": "auto"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, text, instruct, model_choice, device, precision, language,
                 seed=0, max_new_tokens=2048, top_p=0.8, top_k=20,
                 temperature=1.0, repetition_penalty=1.05, attention="auto",
                 unload_model_after_generate=False):
        _set_seed(seed)
        model = _load_model(model_choice, "design", device, precision, attention,
                             unload_model_after_generate)
        wavs, sr = model.generate_voice_design(
            text=text, language=language, instruct=instruct,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, repetition_penalty=repetition_penalty,
        )
        return (_audio_from_wavs(wavs, sr),)


class FB_Qwen3TTSCustomVoice:
    """Qwen3-TTS: generate audio using a predefined speaker voice."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text":     ("STRING", {"multiline": True, "default": "Hello world",
                                        "placeholder": "Enter text to synthesize"}),
                "speaker":  (["Aiden", "Dylan", "Eric", "Ono_anna", "Ryan",
                               "Serena", "Sohee", "Uncle_fu", "Vivian"],
                             {"default": "Ryan"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device":       (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision":    (["bf16", "fp32"], {"default": "bf16"}),
                "language":     (["Auto", "Chinese", "English", "Japanese", "Korean",
                                  "French", "German", "Spanish", "Portuguese", "Russian", "Italian"],
                                 {"default": "Auto"}),
            },
            "optional": {
                "seed":          ("INT", {"default": 0, "min": 0, "max": 2**64 - 1,
                                          "control_after_generate": True}),
                "instruct":      ("STRING", {"multiline": True, "default": "",
                                             "placeholder": "Style instruction (optional)"}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p":          ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k":          ("INT",   {"default": 20, "min": 0, "max": 100}),
                "temperature":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
                "attention":      (["auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                                   {"default": "auto"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, text, speaker, model_choice, device, precision, language,
                 seed=0, instruct="", max_new_tokens=2048, top_p=0.8, top_k=20,
                 temperature=1.0, repetition_penalty=1.05, attention="auto",
                 unload_model_after_generate=False):
        _set_seed(seed)
        model = _load_model(model_choice, "custom", device, precision, attention,
                             unload_model_after_generate)
        wavs, sr = model.generate_custom_voice(
            text=text, language=language, speaker=speaker,
            instruct=instruct if instruct else None,
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, repetition_penalty=repetition_penalty,
        )
        return (_audio_from_wavs(wavs, sr),)


class FB_Qwen3TTSVoiceClone:
    """Qwen3-TTS: clone a voice from reference audio and synthesize text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_text":  ("STRING", {"multiline": True,
                                            "default": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye."}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device":       (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision":    (["bf16", "fp32"], {"default": "bf16"}),
                "language":     (["Auto", "Chinese", "English", "Japanese", "Korean",
                                  "French", "German", "Spanish", "Portuguese", "Russian", "Italian"],
                                 {"default": "Auto"}),
            },
            "optional": {
                "ref_audio":    ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text":     ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Reference audio text (optional)"}),
                "voice_clone_prompt": ("VOICE_CLONE_PROMPT",
                                       {"tooltip": "Reusable voice clone prompt"}),
                "seed":          ("INT", {"default": 0, "min": 0, "max": 2**64 - 1,
                                          "control_after_generate": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p":          ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k":          ("INT",   {"default": 20, "min": 0, "max": 100}),
                "temperature":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
                "x_vector_only":  ("BOOLEAN", {"default": False}),
                "attention":      (["auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                                   {"default": "auto"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    def generate(self, target_text, model_choice, device, precision, language,
                 ref_audio=None, ref_text="", voice_clone_prompt=None,
                 seed=0, max_new_tokens=2048, top_p=0.8, top_k=20,
                 temperature=1.0, repetition_penalty=1.05, x_vector_only=False,
                 attention="auto", unload_model_after_generate=False):
        _set_seed(seed)
        model = _load_model(model_choice, "clone", device, precision, attention,
                             unload_model_after_generate)

        if voice_clone_prompt is not None:
            wavs, sr = model.generate_voice_clone(
                text=target_text, language=language,
                voice_clone_prompt=voice_clone_prompt,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
            )
        elif ref_audio is not None:
            wf, sample_rate = _audio_array(ref_audio)
            wavs, sr = model.generate_voice_clone(
                text=target_text, language=language,
                ref_audio=(wf, sample_rate),
                ref_text=ref_text if (ref_text and not x_vector_only) else None,
                x_vector_only_mode=x_vector_only,
                max_new_tokens=max_new_tokens, temperature=temperature,
                top_p=top_p, repetition_penalty=repetition_penalty,
            )
        else:
            raise ValueError("FB_Qwen3TTSVoiceClone: provide ref_audio or voice_clone_prompt.")

        return (_audio_from_wavs(wavs, sr),)


class FB_Qwen3TTSVoiceClonePrompt:
    """Qwen3-TTS: extract a reusable voice identity prompt from reference audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_audio":    ("AUDIO", {"tooltip": "Reference audio (ComfyUI Audio)"}),
                "ref_text":     ("STRING", {"multiline": True, "default": "",
                                            "placeholder": "Reference audio text (highly recommended)"}),
                "model_choice": (["0.6B", "1.7B"], {"default": "0.6B"}),
                "device":       (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision":    (["bf16", "fp32"], {"default": "bf16"}),
                "attention":    (["auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                                 {"default": "auto"}),
            },
            "optional": {
                "x_vector_only": ("BOOLEAN", {"default": False,
                                              "tooltip": "Speaker embedding only (ref_text not needed)"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("VOICE_CLONE_PROMPT",)
    RETURN_NAMES = ("voice_clone_prompt",)
    FUNCTION = "create_prompt"
    CATEGORY = "Qwen3-TTS"

    def create_prompt(self, ref_audio, ref_text, model_choice, device, precision, attention,
                      x_vector_only=False, unload_model_after_generate=False):
        model = _load_model(model_choice, "clone", device, precision, attention,
                             unload_model_after_generate)
        wf, sr = _audio_array(ref_audio)
        prompt = model.create_voice_clone_prompt(
            ref_audio=(wf, sr),
            ref_text=ref_text if (ref_text and not x_vector_only) else None,
            x_vector_only_mode=x_vector_only,
        )
        return (prompt,)


class FB_Qwen3TTSRoleBank:
    """Qwen3-TTS: collect up to 8 named voice prompts into a role bank for dialogue."""

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for i in range(1, 9):
            optional[f"role_name_{i}"] = ("STRING", {"default": f"Role{i}"})
            optional[f"prompt_{i}"]    = ("VOICE_CLONE_PROMPT",)
        return {"required": {}, "optional": optional}

    RETURN_TYPES = ("QWEN3_ROLE_BANK",)
    RETURN_NAMES = ("role_bank",)
    FUNCTION = "build"
    CATEGORY = "Qwen3-TTS"

    def build(self, **kwargs):
        bank: dict[str, object] = {}
        for i in range(1, 9):
            name   = kwargs.get(f"role_name_{i}", f"Role{i}")
            prompt = kwargs.get(f"prompt_{i}")
            if prompt is not None and name:
                bank[name.strip()] = prompt
        return (bank,)


class FB_Qwen3TTSDialogueInference:
    """Qwen3-TTS: generate multi-character dialogue audio from a script + role bank."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script":    ("STRING", {"multiline": True,
                                         "default": "Role1: Hello, how are you?\nRole2: I am fine, thank you.",
                                         "placeholder": "Format: RoleName: Text"}),
                "role_bank": ("QWEN3_ROLE_BANK",),
                "model_choice": (["0.6B", "1.7B"], {"default": "1.7B"}),
                "device":       (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "precision":    (["bf16", "fp32"], {"default": "bf16"}),
                "language":     (["Auto", "Chinese", "English", "Japanese", "Korean",
                                  "French", "German", "Spanish", "Portuguese", "Russian", "Italian"],
                                 {"default": "Auto"}),
                "pause_linebreak": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                                              "tooltip": "Silence between lines (seconds)"}),
                "period_pause":    ("FLOAT", {"default": 0.4, "min": 0.0, "max": 5.0, "step": 0.1}),
                "comma_pause":     ("FLOAT", {"default": 0.2, "min": 0.0, "max": 5.0, "step": 0.1}),
                "question_pause":  ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.1}),
                "hyphen_pause":    ("FLOAT", {"default": 0.3, "min": 0.0, "max": 5.0, "step": 0.1}),
                "merge_outputs":   ("BOOLEAN", {"default": True,
                                                "tooltip": "Merge all segments into one audio"}),
                "batch_size":      ("INT", {"default": 4, "min": 1, "max": 32,
                                            "tooltip": "Lines processed in parallel"}),
            },
            "optional": {
                "seed":          ("INT", {"default": 0, "min": 0, "max": 2**64 - 1,
                                          "control_after_generate": True}),
                "max_new_tokens_per_line": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 256}),
                "top_p":          ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "top_k":          ("INT",   {"default": 20, "min": 0, "max": 100}),
                "temperature":    ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "repetition_penalty": ("FLOAT", {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.05}),
                "attention":      (["auto", "sage_attn", "flash_attn", "sdpa", "eager"],
                                   {"default": "auto"}),
                "unload_model_after_generate": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS"

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_script(script: str) -> list[tuple[str, str]]:
        """Parse "RoleName: text" lines. Returns [(role, text), ...]."""
        lines = []
        for raw in script.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            m = re.match(r"^([^:]+):\s*(.+)$", raw)
            if m:
                lines.append((m.group(1).strip(), m.group(2).strip()))
            else:
                # No speaker prefix — attribute to previous speaker or skip
                if lines:
                    prev_role, prev_text = lines[-1]
                    lines[-1] = (prev_role, prev_text + " " + raw)
        return lines

    @staticmethod
    def _add_punctuation_pauses(wav: np.ndarray, text: str, sr: int,
                                 period: float, comma: float,
                                 question: float, hyphen: float) -> np.ndarray:
        """Naively append silence based on trailing punctuation in text."""
        text = text.rstrip()
        if text.endswith("?"):
            return np.concatenate([wav, _silence_samples(question, sr)])
        if text.endswith(".") or text.endswith("!"):
            return np.concatenate([wav, _silence_samples(period, sr)])
        if text.endswith(","):
            return np.concatenate([wav, _silence_samples(comma, sr)])
        if text.endswith("-"):
            return np.concatenate([wav, _silence_samples(hyphen, sr)])
        return wav

    # ------------------------------------------------------------------
    def generate(self, script, role_bank, model_choice, device, precision, language,
                 pause_linebreak=0.5, period_pause=0.4, comma_pause=0.2,
                 question_pause=0.6, hyphen_pause=0.3, merge_outputs=True,
                 batch_size=4, seed=0, max_new_tokens_per_line=2048,
                 top_p=0.8, top_k=20, temperature=1.0, repetition_penalty=1.05,
                 attention="auto", unload_model_after_generate=False):

        _set_seed(seed)
        model = _load_model(model_choice, "clone", device, precision, attention, False)

        lines = self._parse_script(script)
        if not lines:
            raise ValueError("FB_Qwen3TTSDialogueInference: script is empty or unparseable.")

        segments: list[np.ndarray] = []
        sr_out: int = 24000

        for role, text in lines:
            prompt = role_bank.get(role)
            if prompt is None:
                print(f"[FB_Qwen3TTS] WARNING: role '{role}' not in role bank — skipping line.")
                continue

            wavs, sr_out = model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=prompt,
                max_new_tokens=max_new_tokens_per_line,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            seg = wavs[0]
            seg = self._add_punctuation_pauses(seg, text, sr_out,
                                               period_pause, comma_pause,
                                               question_pause, hyphen_pause)
            segments.append(seg)
            # Line break pause
            if pause_linebreak > 0:
                segments.append(_silence_samples(pause_linebreak, sr_out))

        if not segments:
            raise RuntimeError("FB_Qwen3TTSDialogueInference: no audio segments generated.")

        if merge_outputs:
            merged = np.concatenate(segments)
            waveform = torch.from_numpy(merged).unsqueeze(0).unsqueeze(0)
            audio = {"waveform": waveform, "sample_rate": sr_out}
        else:
            # Return first segment only (ComfyUI AUDIO is single-track)
            waveform = torch.from_numpy(segments[0]).unsqueeze(0).unsqueeze(0)
            audio = {"waveform": waveform, "sample_rate": sr_out}

        if unload_model_after_generate:
            # Clear this model from cache
            for k in list(_MODEL_CACHE.keys()):
                del _MODEL_CACHE[k]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return (audio,)


# ---------------------------------------------------------------------------
# Exports — these are merged into __init__.py's NODE_CLASS_MAPPINGS
# ---------------------------------------------------------------------------

FB_NODE_CLASS_MAPPINGS = {
    "FB_Qwen3TTSVoiceDesign":       FB_Qwen3TTSVoiceDesign,
    "FB_Qwen3TTSCustomVoice":       FB_Qwen3TTSCustomVoice,
    "FB_Qwen3TTSVoiceClone":        FB_Qwen3TTSVoiceClone,
    "FB_Qwen3TTSVoiceClonePrompt":  FB_Qwen3TTSVoiceClonePrompt,
    "FB_Qwen3TTSRoleBank":          FB_Qwen3TTSRoleBank,
    "FB_Qwen3TTSDialogueInference": FB_Qwen3TTSDialogueInference,
}

FB_NODE_DISPLAY_NAME_MAPPINGS = {
    "FB_Qwen3TTSVoiceDesign":       "Qwen3 TTS Voice Design [Cloud]",
    "FB_Qwen3TTSCustomVoice":       "Qwen3 TTS Custom Voice [Cloud]",
    "FB_Qwen3TTSVoiceClone":        "Qwen3 TTS Voice Clone [Cloud]",
    "FB_Qwen3TTSVoiceClonePrompt":  "Qwen3 TTS Voice Clone Prompt [Cloud]",
    "FB_Qwen3TTSRoleBank":          "Qwen3 TTS Role Bank [Cloud]",
    "FB_Qwen3TTSDialogueInference": "Qwen3 TTS Dialogue Inference [Cloud]",
}
