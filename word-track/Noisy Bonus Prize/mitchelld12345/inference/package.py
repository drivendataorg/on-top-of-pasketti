"""Weight extraction and manifest generation for Qwen3-ASR submission packaging."""
import shutil
import sys
from pathlib import Path

import torch

SUBMISSION_DIR = Path(__file__).parent
PROJECT_ROOT = SUBMISSION_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def save_qwen_model(model_name, checkpoint_path=None):
    from qwen_asr import Qwen3ASRModel

    print(f"Loading pretrained {model_name} on CPU...")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_name, device_map="cpu", max_new_tokens=2048, dtype=torch.bfloat16,
    )
    thinker = wrapper.model.thinker
    processor = wrapper.processor
    del wrapper

    if checkpoint_path and checkpoint_path != "none":
        print(f"Loading finetuned weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        del ckpt
        thinker_sd = {}
        for k, v in state_dict.items():
            if k.startswith("_base_thinker."):
                continue
            if k.startswith("thinker."):
                thinker_sd[k[len("thinker."):]] = v
            else:
                thinker_sd[k] = v
        missing, unexpected = thinker.load_state_dict(thinker_sd, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys")
        print(f"Loaded {len(thinker_sd)} params into thinker")

    if hasattr(thinker, "generation_config") and thinker.generation_config is not None:
        thinker.generation_config.temperature = None

    save_dir = SUBMISSION_DIR / "qwen_model"
    if save_dir.exists():
        shutil.rmtree(save_dir)

    thinker.save_pretrained(str(save_dir))
    processor.save_pretrained(str(save_dir))
    print(f"Saved Qwen model to {save_dir}")
    return save_dir


def save_qwen_full_model(model_name, checkpoint_path=None):
    from qwen_asr import Qwen3ASRModel

    print(f"Loading pretrained {model_name} on CPU (full model)...")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_name, device_map="cpu", max_new_tokens=2048, dtype=torch.bfloat16,
    )
    full_model = wrapper.model
    processor = wrapper.processor
    del wrapper

    if checkpoint_path and checkpoint_path != "none":
        print(f"Loading finetuned thinker weights from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"]
        del ckpt
        thinker_sd = {}
        for k, v in state_dict.items():
            if k.startswith("_base_thinker."):
                continue
            if k.startswith("thinker."):
                thinker_sd[k[len("thinker."):]] = v
            else:
                thinker_sd[k] = v
        missing, unexpected = full_model.thinker.load_state_dict(thinker_sd, strict=False)
        if missing:
            print(f"Warning: {len(missing)} missing keys")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys")
        print(f"Loaded {len(thinker_sd)} params into thinker")

    full_model.thinker = full_model.thinker.to(torch.bfloat16)
    full_model.generation_config.temperature = 1.0

    save_dir = SUBMISSION_DIR / "qwen_model"
    if save_dir.exists():
        shutil.rmtree(save_dir)

    full_model.save_pretrained(str(save_dir))
    processor.save_pretrained(str(save_dir))
    print(f"Saved full Qwen model to {save_dir}")
    return save_dir


def package_qwen_word(checkpoint_path):
    model_name = "Qwen/Qwen3-ASR-1.7B"
    for arg in sys.argv:
        if arg.startswith("--model-name="):
            model_name = arg.split("=", 1)[1]
    if "--full-model" in sys.argv:
        save_qwen_full_model(model_name, checkpoint_path)
    else:
        save_qwen_model(model_name, checkpoint_path)
    return ["main.py", "config.yaml", "silero_vad/", "qwen_model/"]


def main():
    if len(sys.argv) < 4:
        print("Usage: python submission/package.py <model_type> <track> <checkpoint> [--model-name=...] [--full-model]")
        sys.exit(1)

    model_type, track, checkpoint = sys.argv[1], sys.argv[2], sys.argv[3]

    if model_type != "qwen" or track != "word":
        print(f"Error: only qwen/word is supported, got {model_type}/{track}")
        sys.exit(1)

    zip_files = package_qwen_word(checkpoint)

    import datetime, subprocess
    git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    version = f"{timestamp} {git_hash}"
    version_path = SUBMISSION_DIR / "VERSION"
    version_path.write_text(version + "\n")
    zip_files.append("VERSION")
    print(f"Version: {version}")

    manifest_path = SUBMISSION_DIR / ".zip_manifest"
    manifest_path.write_text("\n".join(zip_files) + "\n")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
