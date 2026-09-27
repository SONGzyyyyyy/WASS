import os
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from utils import (
    _create_model_training_folder,
    DataAugment
)

from loss import (
    supervised_contrastive_loss,
    classification_loss,
    wass_center_loss,
    build_sinkhorn_loss
)


RANDOM_SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(RANDOM_SEED)


class WASSTrainer:

    def __init__(
            self,
            online_network,
            target_network,
            optimizer,
            device,
            **params
    ):

        self.online_network = online_network
        self.target_network = target_network

        self.optimizer = optimizer
        self.device = device

        # =================================================
        # Training parameters
        # =================================================

        self.max_epochs = params["max_epochs"]
        self.batch_size = params["batch_size"]

        # EMA coefficient
        self.m = params.get(
            "m",
            0.996
        )

        # =================================================
        # Loss parameters
        # =================================================

        self.temperature = params.get(
            "temperature",
            0.07
        )

        self.lambda_scl = params.get(
            "lambda_scl",
            0.1
        )

        self.lambda_wass = params.get(
            "lambda_wass",
            0.1
        )

        # =================================================
        # Wasserstein loss mode
        # =================================================
        #
        # sinkhorn:
        #   Multi-support-point Wasserstein regularization
        #
        # prototype:
        #   Original prototype-based regularization
        #

        self.wass_mode = params.get(
            "wass_mode",
            "sinkhorn"
        ).lower()

        if self.wass_mode not in {
            "sinkhorn",
            "prototype"
        }:
            raise ValueError(
                f"Unsupported wass_mode: {self.wass_mode}. "
                f"Choose 'sinkhorn' or 'prototype'."
            )

        # =================================================
        # Sinkhorn loss
        # =================================================

        self.sinkhorn_loss = None

        if self.wass_mode == "sinkhorn":

            self.sinkhorn_loss = build_sinkhorn_loss(
                blur=params.get(
                    "sinkhorn_blur",
                    0.05
                ),
                p=params.get(
                    "sinkhorn_p",
                    2
                )
            )

        # =================================================
        # Early stopping
        # =================================================

        self.patience = params.get(
            "patience",
            10
        )

        self.min_delta = params.get(
            "min_delta",
            0.001
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # =================================================
        # Logging
        # =================================================

        run_name = params.get(
            "run_name",
            "runs/seed42_wass"
        )

        self.writer = SummaryWriter(
            run_name
        )

        os.makedirs(
            "Base_trainer",
            exist_ok=True
        )

        _create_model_training_folder(
            self.writer,
            files_to_same=[
                "./config/config.yaml",
                "main.py",
                "trainer.py",
                "loss/supervised_contrastive_loss.py",
                "loss/wass_center_loss.py",
                "loss/classification_loss.py"
            ]
        )

        print(
            f"WASS mode: {self.wass_mode}"
        )

    # =====================================================
    # Target network
    # =====================================================

    @torch.no_grad()
    def update_target_network(self):

        for online_param, target_param in zip(
                self.online_network.parameters(),
                self.target_network.parameters()
        ):

            target_param.data.mul_(
                self.m
            )

            target_param.data.add_(
                online_param.data *
                (1.0 - self.m)
            )

    @torch.no_grad()
    def initialize_target_network(self):

        for online_param, target_param in zip(
                self.online_network.parameters(),
                self.target_network.parameters()
        ):

            target_param.data.copy_(
                online_param.data
            )

        # Target network is not optimized by gradient descent.
        for target_param in self.target_network.parameters():

            target_param.requires_grad = False

        # Important:
        # keep BatchNorm statistics of the target network fixed.
        self.target_network.eval()

    # =====================================================
    # Training
    # =====================================================

    def train(
            self,
            train_loader,
            epoch
    ):

        self.online_network.train()

        # Momentum/target encoder should remain in eval mode.
        self.target_network.eval()

        augment = DataAugment()

        total_loss = 0.0
        total_cls = 0.0
        total_scl = 0.0
        total_wass = 0.0

        num_batches = len(train_loader)

        if num_batches == 0:
            raise RuntimeError(
                "Training DataLoader is empty."
            )

        for (
            batch_view,
            labels,
            domains
        ) in train_loader:

            batch_view = batch_view.to(
                self.device,
                non_blocking=True
            )

            labels = labels.long().to(
                self.device,
                non_blocking=True
            )

            domains = domains.long().to(
                self.device,
                non_blocking=True
            )

            # ---------------------------------------------
            # Two augmented views
            # ---------------------------------------------

            view_1 = augment(
                batch_view
            )

            view_2 = augment(
                batch_view
            )

            self.optimizer.zero_grad(
                set_to_none=True
            )

            (
                loss,
                cls_loss,
                scl_loss,
                wass_loss
            ) = self.compute_loss(
                view_0=batch_view,
                view_1=view_1,
                view_2=view_2,
                labels=labels,
                domains=domains
            )

            loss.backward()

            self.optimizer.step()

            # EMA update after optimizer step.
            self.update_target_network()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_scl += scl_loss.item()
            total_wass += wass_loss.item()

        total_loss /= num_batches
        total_cls /= num_batches
        total_scl /= num_batches
        total_wass /= num_batches

        # =================================================
        # TensorBoard
        # =================================================

        self.writer.add_scalar(
            "train/total_loss",
            total_loss,
            epoch
        )

        self.writer.add_scalar(
            "train/classification_loss",
            total_cls,
            epoch
        )

        self.writer.add_scalar(
            "train/scl_loss",
            total_scl,
            epoch
        )

        self.writer.add_scalar(
            "train/wass_loss",
            total_wass,
            epoch
        )

        print(
            f"Train: "
            f"total={total_loss:.6f}, "
            f"cls={total_cls:.6f}, "
            f"scl={total_scl:.6f}, "
            f"wass={total_wass:.6f}"
        )

        return total_loss

    # =====================================================
    # Validation
    # =====================================================

    @torch.no_grad()
    def validate(
            self,
            val_loader,
            epoch
    ):

        self.online_network.eval()
        self.target_network.eval()

        total_loss = 0.0

        num_batches = len(val_loader)

        if num_batches == 0:
            raise RuntimeError(
                "Validation DataLoader is empty."
            )

        for (
            batch_view,
            labels,
            _
        ) in val_loader:

            batch_view = batch_view.to(
                self.device,
                non_blocking=True
            )

            labels = labels.long().to(
                self.device,
                non_blocking=True
            )

            (
                _,
                _,
                logits,
                _
            ) = self.online_network(
                batch_view
            )

            loss = classification_loss(
                logits,
                labels
            )

            total_loss += loss.item()

        total_loss /= num_batches

        self.writer.add_scalar(
            "val/classification_loss",
            total_loss,
            epoch
        )

        print(
            f"Validation Loss: "
            f"{total_loss:.6f}"
        )

        return total_loss

    # =====================================================
    # Total loss
    # =====================================================

    def compute_loss(
            self,
            view_0,
            view_1,
            view_2,
            labels,
            domains
    ):

        # =================================================
        # Online network
        # =================================================

        (
            feature_0,
            _,
            logits_0,
            support_points
        ) = self.online_network(
            view_0
        )

        (
            _,
            projection_1,
            _,
            _
        ) = self.online_network(
            view_1
        )

        (
            _,
            projection_2,
            _,
            _
        ) = self.online_network(
            view_2
        )

        # =================================================
        # Target network
        # =================================================

        with torch.no_grad():

            target_projection_1 = (
                self.target_network(
                    view_1
                )[1]
            )

            target_projection_2 = (
                self.target_network(
                    view_2
                )[1]
            )

        # =================================================
        # Supervised contrastive learning
        # =================================================
        #
        # Compute both directions:
        #
        #   online view_1 -> target view_2
        #   online view_2 -> target view_1
        #
        # and average them.
        #

        scl_loss_1 = supervised_contrastive_loss(
            projection_1,
            target_projection_2,
            labels,
            temperature=self.temperature
        )

        scl_loss_2 = supervised_contrastive_loss(
            projection_2,
            target_projection_1,
            labels,
            temperature=self.temperature
        )

        scl_loss = 0.5 * (
            scl_loss_1
            + scl_loss_2
        )

        # =================================================
        # Wasserstein regularization
        # =================================================

        wass_loss = wass_center_loss(
            features=feature_0,
            support_points=support_points,
            labels=labels,
            domains=domains,
            mode=self.wass_mode,
            sinkhorn_loss=self.sinkhorn_loss
        )

        # =================================================
        # Classification loss
        # =================================================

        cls_loss = classification_loss(
            logits_0,
            labels
        )

        # =================================================
        # Total loss
        # =================================================

        total_loss = (
            cls_loss
            + self.lambda_scl * scl_loss
            + self.lambda_wass * wass_loss
        )

        return (
            total_loss,
            cls_loss,
            scl_loss,
            wass_loss
        )

    # =====================================================
    # Train + validation
    # =====================================================

    def train_and_val(
            self,
            train_dataset,
            val_dataset
    ):

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=torch.cuda.is_available()
        )

        # -------------------------------------------------
        # Initialize EMA target network
        # -------------------------------------------------

        self.initialize_target_network()

        train_history = []
        val_history = []

        for epoch in range(
            self.max_epochs
        ):

            print(
                f"\nEpoch "
                f"[{epoch + 1}/{self.max_epochs}]"
            )

            # -------------------------------------------------
            # Training
            # -------------------------------------------------

            train_loss = self.train(
                train_loader,
                epoch
            )

            # -------------------------------------------------
            # Validation
            # -------------------------------------------------

            val_loss = self.validate(
                val_loader,
                epoch
            )

            train_history.append(
                train_loss
            )

            val_history.append(
                val_loss
            )

            # =================================================
            # Best model
            # =================================================

            improvement = (
                self.best_val_loss
                - val_loss
            )

            if improvement > self.min_delta:

                self.best_val_loss = val_loss

                self.patience_counter = 0

                torch.save(
                    self.online_network,
                    "Base_trainer/epoch_best.pth"
                )

                print(
                    f"Best model saved. "
                    f"Val loss = {val_loss:.6f}"
                )

            else:

                self.patience_counter += 1

                print(
                    f"Patience: "
                    f"{self.patience_counter}/"
                    f"{self.patience}"
                )

            # =================================================
            # Periodic checkpoint
            # =================================================
            #
            # Save epoch 0, 50, 100, ...
            #

            if epoch % 50 == 0:

                torch.save(
                    self.online_network,
                    f"Base_trainer/"
                    f"epoch{epoch}.pth"
                )

            # =================================================
            # Early stopping
            # =================================================

            if (
                self.patience_counter
                >= self.patience
            ):

                print(
                    "Early stopping triggered."
                )

                break

        # =====================================================
        # Final model
        # =====================================================

        torch.save(
            self.online_network,
            "Base_trainer/epochend.pth"
        )

        # =====================================================
        # Save loss curves
        # =====================================================

        self.plot_loss(
            train_history,
            val_history
        )

        np.savetxt(
            "Base_trainer/train_loss.txt",
            np.asarray(train_history)
        )

        np.savetxt(
            "Base_trainer/eval_loss.txt",
            np.asarray(val_history)
        )

        self.writer.close()

    # =====================================================
    # Loss curve
    # =====================================================

    @staticmethod
    def plot_loss(
            train_history,
            val_history
    ):

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            train_history,
            label="train loss"
        )

        plt.plot(
            val_history,
            label="validation loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "WASS Training Loss"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.savefig(
            "Base_trainer/loss.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()