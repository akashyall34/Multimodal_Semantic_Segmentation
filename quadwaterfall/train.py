"""
Training script for FuseForm on MCubeS Dataset
Based on: McMillen & Yilmaz, "FuseForm: Multimodal Transformer for Semantic Segmentation", WACVW 2025
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import json

import torch
import torch.nn as nn
import torch.optim as optim

from config import Config, config as default_config
from dataset import build_loaders
from architecture import QuadWaterfall


class PolyLRScheduler:
    """Polynomial learning rate scheduler with warmup"""

    def __init__(self, optimizer, lr_init, lr_max, lr_min, num_epochs, warmup_epochs, poly_power):
        self.optimizer = optimizer
        self.lr_init = lr_init
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.poly_power = poly_power
        self.current_epoch = 0

    def step(self, epoch):
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_init + (self.lr_max - self.lr_init) * (epoch / self.warmup_epochs)
        else:
            # Polynomial decay
            progress = (epoch - self.warmup_epochs) / (self.num_epochs - self.warmup_epochs)
            lr = self.lr_max * ((1 - progress) ** self.poly_power)
            lr = max(lr, self.lr_min)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


class SegmentationMetrics:
    """Compute semantic segmentation metrics"""

    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred, target):
        """
        Args:
            pred: (B, H, W) tensor of class predictions
            target: (B, H, W) tensor of ground truth labels
        """
        pred = pred.cpu().numpy().astype(np.int64)
        target = target.cpu().numpy().astype(np.int64)

        mask = target != self.ignore_index if self.ignore_index is not None else np.ones_like(target, dtype=bool)

        pred = pred[mask]
        target = target[mask]

        self.confusion_matrix += np.bincount(
            self.num_classes * target + pred,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)

    def compute_metrics(self):
        """Compute mIoU, mAccuracy, allAccuracy"""
        iou_per_class = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=0) + np.sum(self.confusion_matrix, axis=1)
            - np.diag(self.confusion_matrix) + 1e-10
        )
        acc_per_class = np.diag(self.confusion_matrix) / (np.sum(self.confusion_matrix, axis=1) + 1e-10)

        miou = np.mean(iou_per_class)
        macc = np.mean(acc_per_class)
        aacc = np.sum(np.diag(self.confusion_matrix)) / (np.sum(self.confusion_matrix) + 1e-10)

        return {
            "mIoU": miou,
            "mAccuracy": macc,
            "allAccuracy": aacc,
            "iou_per_class": iou_per_class,
            "acc_per_class": acc_per_class,
        }


class Trainer:
    """Training loop for semantic segmentation"""

    def __init__(self, cfg: Config, model, device="cuda"):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device

        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = build_loaders(
            root=cfg.data.root,
            crop_size=cfg.data.crop_size,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.data.num_workers,
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.train.lr_max,
            weight_decay=cfg.train.weight_decay,
            eps=cfg.train.adam_epsilon,
        )

        # Learning rate scheduler
        self.scheduler = PolyLRScheduler(
            optimizer=self.optimizer,
            lr_init=cfg.train.lr_init,
            lr_max=cfg.train.lr_max,
            lr_min=cfg.train.lr_min,
            num_epochs=cfg.train.num_epochs,
            warmup_epochs=cfg.train.warmup_epochs,
            poly_power=cfg.train.poly_power,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)

        # Metrics
        self.metrics = SegmentationMetrics(num_classes=cfg.data.num_classes, ignore_index=255)

        # Checkpointing
        self.checkpoint_dir = Path(cfg.train.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training history (saved to /tmp — ephemeral, not persisted to Drive)
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_miou": [], "val_miou": [],
            "train_macc": [], "val_macc": [],
            "train_allacc": [], "val_allacc": [],
            "lr": [],
        }
        self.history_path = "/tmp/training_history.json"

        self.best_miou = 0.0
        self.global_step = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            rgb = batch["rgb"].to(self.device)
            aolp = batch["aolp"].to(self.device)
            dolp = batch["dolp"].to(self.device)
            nir = batch["nir"].to(self.device)
            label = batch["label"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(rgb, aolp, dolp, nir)  # (B, num_classes, H, W)

            # Compute loss
            loss = self.criterion(output, label)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            pred = torch.argmax(output, dim=1)
            self.metrics.update(pred, label)

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{self.cfg.train.num_epochs} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        # Log metrics
        metrics = self.metrics.compute_metrics()
        avg_loss = total_loss / num_batches

        print(f"\nEpoch {epoch + 1} Train Metrics:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {metrics['mIoU']:.4f}")
        print(f"  mAccuracy: {metrics['mAccuracy']:.4f}")
        print(f"  allAccuracy: {metrics['allAccuracy']:.4f}\n")

        # Record history
        self.history["train_loss"].append(avg_loss)
        self.history["train_miou"].append(float(metrics["mIoU"]))
        self.history["train_macc"].append(float(metrics["mAccuracy"]))
        self.history["train_allacc"].append(float(metrics["allAccuracy"]))
        self.history["lr"].append(self.scheduler.get_lr())

        return avg_loss, metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Validate on validation set"""
        self.model.eval()
        self.metrics.reset()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            rgb = batch["rgb"].to(self.device)
            aolp = batch["aolp"].to(self.device)
            dolp = batch["dolp"].to(self.device)
            nir = batch["nir"].to(self.device)
            label = batch["label"].to(self.device)

            # Forward pass
            output = self.model(rgb, aolp, dolp, nir)  # (B, num_classes, H, W)

            loss = self.criterion(output, label)

            pred = torch.argmax(output, dim=1)
            self.metrics.update(pred, label)

            total_loss += loss.item()
            num_batches += 1

        metrics = self.metrics.compute_metrics()
        avg_loss = total_loss / num_batches

        print(f"Epoch {epoch + 1} Val Metrics:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  mIoU: {metrics['mIoU']:.4f}")
        print(f"  mAccuracy: {metrics['mAccuracy']:.4f}")
        print(f"  allAccuracy: {metrics['allAccuracy']:.4f}\n")

        # Record history
        self.history["val_loss"].append(avg_loss)
        self.history["val_miou"].append(float(metrics["mIoU"]))
        self.history["val_macc"].append(float(metrics["mAccuracy"]))
        self.history["val_allacc"].append(float(metrics["allAccuracy"]))

        return avg_loss, metrics

    @torch.no_grad()
    def test(self):
        """Test on test set"""
        self.model.eval()
        self.metrics.reset()

        print("Running test evaluation...")

        for batch in self.test_loader:
            rgb = batch["rgb"].to(self.device)
            aolp = batch["aolp"].to(self.device)
            dolp = batch["dolp"].to(self.device)
            nir = batch["nir"].to(self.device)
            label = batch["label"].to(self.device)

            output = self.model(rgb, aolp, dolp, nir)  # (B, num_classes, H, W)
            pred = torch.argmax(output, dim=1)
            self.metrics.update(pred, label)

        metrics = self.metrics.compute_metrics()

        print("\nTest Results:")
        print(f"  mIoU: {metrics['mIoU']:.4f}")
        print(f"  mAccuracy: {metrics['mAccuracy']:.4f}")
        print(f"  allAccuracy: {metrics['allAccuracy']:.4f}\n")

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }

        # Save periodic checkpoint
        if (epoch + 1) % self.cfg.train.save_interval == 0:
            ckpt_path = self.checkpoint_dir / f"epoch_{epoch + 1}.pt"
            torch.save(checkpoint, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Checkpoint loaded: {checkpoint_path}")
        return ckpt["epoch"]

    def train(self):
        """Full training loop"""
        print(f"Starting training for {self.cfg.train.num_epochs} epochs...")
        print(f"Warmup: {self.cfg.train.warmup_epochs} epochs")
        print(f"LR: {self.cfg.train.lr_init} → {self.cfg.train.lr_max} → {self.cfg.train.lr_min}")
        print(f"Poly power: {self.cfg.train.poly_power}\n")

        for epoch in range(self.cfg.train.num_epochs):
            # Train
            self.train_epoch(epoch)

            # Update learning rate
            self.scheduler.step(epoch)

            # Validate
            if (epoch + 1) % self.cfg.eval.val_interval == 0:
                val_loss, val_metrics = self.validate(epoch)

                # Save checkpoint
                is_best = val_metrics["mIoU"] > self.best_miou
                if is_best:
                    self.best_miou = val_metrics["mIoU"]

                if self.cfg.train.save_best:
                    self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Persist history after each epoch
            with open(self.history_path, "w") as f:
                json.dump(self.history, f)

        # Reload best checkpoint before test evaluation
        best_ckpt_path = self.checkpoint_dir / "best_model.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best checkpoint (epoch {ckpt['epoch'] + 1}, val mIoU: {ckpt['metrics']['mIoU']:.4f}) for test evaluation")
        else:
            print("Warning: best_model.pt not found; evaluating with final epoch weights")

        # Test
        self.test()

        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train FuseForm on MCubeS dataset")
    parser.add_argument("--data-root", type=str, default="../multimodal_dataset", help="Path to MCubeS dataset")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=6e-6, help="Maximum learning rate")
    parser.add_argument("--lr-init", type=float, default=None, help="Initial learning rate for warmup")
    parser.add_argument("--lr-max", type=float, default=None, help="Maximum learning rate after warmup")
    parser.add_argument("--lr-min", type=float, default=None, help="Minimum learning rate")
    parser.add_argument("--warmup-epochs", type=int, default=None, help="Number of warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=None, help="Weight decay for AdamW")
    parser.add_argument("--poly-power", type=float, default=None, help="Polynomial decay power")
    parser.add_argument("--dropout-p", type=float, default=None, help="Dropout probability")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Create fresh config (not using global default to avoid parameter persistence)
    cfg = Config()
    cfg.data.root = args.data_root
    cfg.train.batch_size = args.batch_size
    cfg.train.num_epochs = args.num_epochs
    cfg.train.device = args.device

    # Learning rate parameters (use provided values or defaults)
    cfg.train.lr_max = args.lr_max if args.lr_max is not None else args.lr
    cfg.train.lr_init = args.lr_init if args.lr_init is not None else cfg.train.lr_init
    cfg.train.lr_min = args.lr_min if args.lr_min is not None else cfg.train.lr_min
    cfg.train.warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else cfg.train.warmup_epochs

    # Optimizer parameters
    cfg.train.weight_decay = args.weight_decay if args.weight_decay is not None else cfg.train.weight_decay
    cfg.train.poly_power = args.poly_power if args.poly_power is not None else cfg.train.poly_power

    # Model dropout
    dropout_p = args.dropout_p if args.dropout_p is not None else 0.1

    # Initialize QuadWaterfall model
    model = QuadWaterfall(
        num_classes=cfg.data.num_classes,
        rgb_var='b4',           # MiT-B4 for RGB
        aux_var='b2',           # MiT-B2 for AoLP/DoLP/NIR
        pretrained=True,        # Load ImageNet pretrained weights
        p=dropout_p,            # Dropout rate
        enc_checkpoint=True,    # Gradient checkpointing for memory efficiency
        qwtm_checkpoint=True,   # Gradient checkpointing for QWTM
    )

    # Create trainer
    trainer = Trainer(cfg, model, device=args.device)

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
