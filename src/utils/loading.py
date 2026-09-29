"""Utility functions for testing."""

# %%
import tempfile
from copy import deepcopy
from pathlib import Path

import torch
from lightning import LightningModule
from loguru import logger


def custom_load(checkpoint: Path, module: LightningModule) -> LightningModule:
    """Load the model from the checkpoint.

    This function brought to you by the normal checkpoint restoring not working as advertised.
    (╯°□°)╯︵ ┻━┻

    Parameters
    ----------
    checkpoint : Path
        The path to the checkpoint.
    module : LightningModule
        The module to load the checkpoint into.

    Returns
    -------
    LightningModule
        The loaded module.

    """  # noqa: E501
    chkpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = chkpt["state_dict"]

    model = module(**chkpt["hyper_parameters"])
    model.load_state_dict(strip_state_dict(state_dict))
    return model


def sanitize_ckpt_path(ckpt_path: str | None) -> str | None:
    """Remove `_orig_mod.` prefix from checkpoint due to `model.compile`.

    Parameters
    ----------
    ckpt_path : str | None
        The path to the checkpoint.

    Returns
    -------
    str | None
        The path to the patched checkpoint, or the original path if no patching was needed.

    """  # noqa: E501
    if ckpt_path is None:
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", {})

    if not any("._orig_mod." in k for k in state_dict):
        return ckpt_path

    logger.info(
        "Detected torch.compiled checkpoint, stripping `_orig_mod.` from keys",
    )

    ckpt["state_dict"] = strip_state_dict(state_dict)

    tmp_path = Path(tempfile.gettempdir()) / f"sanitized_{Path(ckpt_path).name}"
    torch.save(ckpt, tmp_path)
    return str(tmp_path)


def strip_state_dict(state_dict: dict) -> dict:
    """Remove `_orig_mod.` prefix from the state_dict due to `model.compile`.

    Parameters
    ----------
    state_dict : dict
        The state_dict to patch.

    Returns
    -------
    dict
        The stripped state_dict.

    """
    # ? Needs to be a deepcopy otherwise RuntimeError
    new_state_dict = deepcopy(state_dict)

    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            new_key = k.replace("model._orig_mod.", "model.")
            new_state_dict[new_key] = v
            del new_state_dict[k]
    return new_state_dict
