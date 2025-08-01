"""Test training and evaluation configurations."""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    check_config(cfg_train)

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    check_config(cfg_eval)

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def check_config(cfg: DictConfig) -> None:
    """Check if the configuration is valid.

    :param cfg: A DictConfig containing a valid training configuration.
    """
    if not cfg:
        msg = "Training configuration is missing"
        raise ValueError(msg)

    if not cfg.data:
        msg = "Data configuration is missing in training config"
        raise ValueError(msg)

    if not cfg.model:
        msg = "Model configuration is missing in training config"
        raise ValueError(msg)

    if not cfg.trainer:
        msg = "Trainer configuration is missing in training config"
        raise ValueError(msg)
