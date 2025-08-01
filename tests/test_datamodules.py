"""Test if the datamodules can be instantiated and data can be loaded correctly."""

from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:  # noqa: C901
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """  # noqa: E501
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    if dm.data_train or dm.data_val or dm.data_test:
        pytest.fail("Dataset vars set prematurely")

    if not Path(data_dir, "MNIST").exists():
        pytest.fail("MNIST directory not created")

    if not Path(data_dir, "MNIST", "raw").exists():
        pytest.fail("Data not downloaded correctly")

    dm.setup()

    if not dm.data_train or not dm.data_val or not dm.data_test:
        pytest.fail("Dataset vars not set correctly")

    if not dm.train_dataloader() or not dm.val_dataloader() or not dm.test_dataloader():
        pytest.fail("Dataloaders not created correctly")

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    correct_num_datapoints = 70_000
    if num_datapoints != correct_num_datapoints:
        pytest.fail(
            f"Number of datapoints is incorrect. Expected {correct_num_datapoints}, got {num_datapoints}",  # noqa: E501
        )

    batch = next(iter(dm.train_dataloader()))
    x, y = batch

    if len(x) != batch_size:
        pytest.fail("Train batch size is incorrect")
    if len(y) != batch_size:
        pytest.fail("Test batch size is incorrect")
    if x.dtype != torch.float32:
        pytest.fail("Train dtype is incorrect")
    if y.dtype != torch.int64:
        pytest.fail("Test dtype is incorrect")
