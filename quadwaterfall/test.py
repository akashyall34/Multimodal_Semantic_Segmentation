"""
Standalone evaluation script using best_model.pt checkpoint on test dataset
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np

from config import Config
from architecture import QuadWaterfall
from dataset import build_loaders


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


def evaluate(checkpoint_path, data_root, device="cuda", batch_size=1, num_workers=4):
    """
    Load best model checkpoint and evaluate on test dataset
    
    Args:
        checkpoint_path: Path to best_model.pt
        data_root: Path to dataset root
        device: Device to use (cuda/cpu)
        batch_size: Batch size for evaluation
        num_workers: Number of workers for data loading
    
    Returns:
        dict: Evaluation metrics
    """
    
    # Check if checkpoint exists
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📁 Loading checkpoint: {checkpoint_path}")
    
    # Create config
    cfg = Config()
    cfg.data.root = data_root
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print(f"✅ Checkpoint loaded (epoch {checkpoint['epoch'] + 1})")
    print(f"   Training mIoU at checkpoint: {checkpoint['metrics'].get('mIoU', 'N/A'):.4f}" if isinstance(checkpoint['metrics'].get('mIoU'), float) else "   (metrics info not available)")
    
    # Initialize model
    model = QuadWaterfall(
        num_classes=cfg.data.num_classes,
        rgb_var='b4',
        aux_var='b2',
        pretrained=False,
        p=0.1,
        enc_checkpoint=True,
        qwtm_checkpoint=True,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded on {device}")
    
    # Build test loader only
    _, _, test_loader = build_loaders(
        root=data_root,
        crop_size=cfg.data.crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    print(f"✅ Test dataset loaded ({len(test_loader.dataset)} samples)")
    
    # Metrics
    metrics_calculator = SegmentationMetrics(num_classes=cfg.data.num_classes, ignore_index=255)
    
    # Evaluation loop
    print("\n🔄 Running evaluation on test set...\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            rgb = batch["rgb"].to(device)
            aolp = batch["aolp"].to(device)
            dolp = batch["dolp"].to(device)
            nir = batch["nir"].to(device)
            label = batch["label"].to(device)
            
            # Forward pass
            output = model(rgb, aolp, dolp, nir)  # (B, num_classes, H, W)
            pred = torch.argmax(output, dim=1)
            
            # Update metrics
            metrics_calculator.update(pred, label)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Compute and print metrics
    metrics = metrics_calculator.compute_metrics()
    
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"  mIoU:           {metrics['mIoU']*100:.2f}%")
    print(f"  mAccuracy:      {metrics['mAccuracy']*100:.2f}%")
    print(f"  Pixel Accuracy: {metrics['allAccuracy']*100:.2f}%")
    print("="*50 + "\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model on test dataset")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best_model.pt",
        help="Path to best_model.pt checkpoint"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../multimodal_dataset",
        help="Path to MCubeS dataset"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for data loading"
    )
    
    args = parser.parse_args()
    
    # Verify device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Run evaluation
    metrics = evaluate(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    if metrics is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
