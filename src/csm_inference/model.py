"""Device selection, model access, and checkpoint downloads."""

from .config import MODEL_ID


def resolve_device(requested="auto"):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in ("mps", "cpu", "cuda"):
        raise ValueError(f"Unknown device: {requested}")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use a native Apple Silicon terminal or --device cpu.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable on this machine.")
    return requested


def check_access():
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError

    try:
        HfApi().auth_check(repo_id=MODEL_ID)
    except HfHubHTTPError as error:
        if getattr(error.response, "status_code", None) in (401, 403):
            raise ValueError(
                f"Access denied for {MODEL_ID}. Accept access at https://huggingface.co/{MODEL_ID} "
                "and run `uv run hf auth login` with a read token for that account."
            ) from None
        raise


def download_model():
    from huggingface_hub import snapshot_download

    check_access()
    return snapshot_download(
        MODEL_ID,
        allow_patterns=[
            "*.json",
            "*.jinja",
            "tokenizer*",
            "transformers*.safetensors",
        ],
    )
