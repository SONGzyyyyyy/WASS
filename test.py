import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F
import yaml

from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from get_dataset import TestDataset_prepared


RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    # Ensure reproducible test results.
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


@torch.no_grad()
def test(
        online_network,
        test_dataloader,
        device
):
    online_network = online_network.to(device)
    online_network.eval()

    test_loss = 0.0
    correct = 0

    criterion = nn.NLLLoss()

    for data, target in test_dataloader:

        data = data.to(
            device,
            non_blocking=True
        )

        target = target.long().to(
            device,
            non_blocking=True
        )

        logits = online_network(data)[2]

        output = F.log_softmax(
            logits,
            dim=1
        )

        test_loss += criterion(
            output,
            target
        ).item()

        pred = output.argmax(
            dim=1
        )

        correct += (
            pred == target
        ).sum().item()

    num_samples = len(
        test_dataloader.dataset
    )

    num_batches = len(
        test_dataloader
    )

    test_loss /= num_batches

    accuracy = (
        100.0
        * correct
        / num_samples
    )

    print(
        "\nTest set: "
        f"Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{num_samples} "
        f"({accuracy:.4f}%)"
    )

    return accuracy


def WASSTest():

    # Load test configuration.
    with open(
            "config/config.yaml",
            "r"
    ) as f:
        config = yaml.safe_load(f)

    test_batch_size = config.get(
        "finetune",
        {}
    ).get(
        "test_batch_size",
        128
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Testing with: {device}"
    )

    # Load the test dataset.
    X_test, Y_test = TestDataset_prepared()

    test_dataset = TensorDataset(
        torch.as_tensor(
            X_test,
            dtype=torch.float32
        ),
        torch.as_tensor(
            Y_test,
            dtype=torch.long
        )
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available()
    )

    # -------------------------------------------------
    # Load the final model
    # -------------------------------------------------

    end_model = torch.load(
        "Base_trainer/epochend.pth",
        map_location=device,
        weights_only=False
    )

    end_test_acc = test(
        end_model,
        test_dataloader,
        device
    )

    print(
        "-" * 100
    )

    print(
        f"End Test Accuracy: "
        f"{end_test_acc:.3f}%"
    )

    # -------------------------------------------------
    # Load the best model
    # -------------------------------------------------

    best_model = torch.load(
        "Base_trainer/epoch_best.pth",
        map_location=device,
        weights_only=False
    )

    best_test_acc = test(
        best_model,
        test_dataloader,
        device
    )

    print(
        "-" * 100
    )

    print(
        f"Best Test Accuracy: "
        f"{best_test_acc:.3f}%"
    )

    # -------------------------------------------------
    # Save test results
    # -------------------------------------------------

    os.makedirs(
        "test_result",
        exist_ok=True
    )

    end_acc_str = f"{end_test_acc:.3f}"
    best_acc_str = f"{best_test_acc:.3f}"

    filename = (
        "test_result/"
        f"END_{end_acc_str}_"
        f"BEST_{best_acc_str}.txt"
    )

    with open(
            filename,
            "w"
    ) as f:

        f.write(
            "WASS test results:\n"
        )

        f.write(
            f"Test Accuracy at the END: "
            f"{end_acc_str}%\n"
        )

        f.write(
            f"Test Accuracy at the BEST: "
            f"{best_acc_str}%\n"
        )

    print(
        f"Test results saved to: "
        f"{filename}"
    )


if __name__ == "__main__":
    WASSTest()