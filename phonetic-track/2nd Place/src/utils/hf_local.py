from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_hf_load_path(model_id: str, inference: bool) -> tuple[str, bool]:
    """Resolve a local Hugging Face asset path for offline inference."""
    if not inference:
        return model_id, False

    local_model_path = PROJECT_ROOT / "external" / model_id.split("/")[-1]
    if local_model_path.exists():
        return str(local_model_path), True

    # In inference mode we should not attempt network access. If the external
    # directory is missing, allow Transformers to check only its local cache.
    return model_id, True
