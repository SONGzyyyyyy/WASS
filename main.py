import os
import random
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import yaml

from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

from models.wass_model import WASSModel
from trainer import WASSTrainer
from get_dataset import PreTrainDataset_prepared
from test import WASSTest


RANDOM_SEED = 42


def set_seed(seed=RANDOM_SEED):
    # Ensure reproducible data splitting and model training.
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    set_seed(RANDOM_SEED)

    # Load model and training configurations.
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    network_config = config["network"]
    trainer_config = config["trainer"]

    # Select GPU when CUDA is available.
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training with: {device}")

    # Load the preprocessed signal samples, class labels,
    # and receiver/domain labels.
    X, Y, Z = PreTrainDataset_prepared()

    # Split the dataset into training and validation sets.
    (
        X_train,
        X_val,
        Y_train,
        Y_val,
        Z_train,
        Z_val
    ) = train_test_split(
        X,
        Y,
        Z,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=Y
    )

    train_dataset = TensorDataset(
        torch.as_tensor(
            X_train,
            dtype=torch.float32
        ),
        torch.as_tensor(
            Y_train,
            dtype=torch.long
        ),
        torch.as_tensor(
            Z_train,
            dtype=torch.long
        )
    )

    val_dataset = TensorDataset(
        torch.as_tensor(
            X_val,
            dtype=torch.float32
        ),
        torch.as_tensor(
            Y_val,
            dtype=torch.long
        ),
        torch.as_tensor(
            Z_val,
            dtype=torch.long
        )
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Build the online and target networks.
    online_network = WASSModel(
        **network_config
    ).to(device)

    target_network = WASSModel(
        **network_config
    ).to(device)

    # Optimize only the online network.
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=trainer_config["lr"]
    )

    # Initialize the WASS trainer.
    trainer = WASSTrainer(
        online_network=online_network,
        target_network=target_network,
        optimizer=optimizer,
        device=device,
        **trainer_config
    )

    # Train the model and perform validation.
    trainer.train_and_val(
        train_dataset,
        val_dataset
    )

    # Evaluate the trained model.
    WASSTest()


if __name__ == "__main__":
    main()