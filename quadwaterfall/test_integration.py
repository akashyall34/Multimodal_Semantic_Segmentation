"""
Integration test: QuadWaterfall + Dataset + Training loop
Verifies data shapes, model forward pass, and training step
"""

import torch
from architecture import QuadWaterfall
from dataset import MCubeSDataset, build_loaders
from config import config

def test_model_forward():
    """Test model forward pass"""
    print("=" * 60)
    print("TEST 1: Model Forward Pass")
    print("=" * 60)

    B, H, W = 2, 512, 512
    model = QuadWaterfall(num_classes=20, pretrained=False).eval()

    rgb = torch.randn(B, 3, H, W)
    aolp = torch.randn(B, 1, H, W)
    dolp = torch.randn(B, 1, H, W)
    nir = torch.randn(B, 1, H, W)

    print(f"Input shapes:")
    print(f"  RGB:  {tuple(rgb.shape)}")
    print(f"  AoLP: {tuple(aolp.shape)}")
    print(f"  DoLP: {tuple(dolp.shape)}")
    print(f"  NIR:  {tuple(nir.shape)}")

    with torch.no_grad():
        output = model(rgb, aolp, dolp, nir)

    print(f"\nOutput shape: {tuple(output.shape)}")
    print(f"Expected:     ({B}, 20, {H}, {W})")

    assert output.shape == (B, 20, H, W), f"Shape mismatch! Got {output.shape}"
    print("\n✅ Model forward pass: PASSED\n")


def test_dataset_loading(data_root="../multimodal_dataset"):
    """Test dataset loading"""
    print("=" * 60)
    print("TEST 2: Dataset Loading")
    print("=" * 60)

    try:
        ds = MCubeSDataset(root=data_root, split='train', crop_size=512)
        print(f"Dataset loaded successfully")
        print(f"  Split: train")
        print(f"  Samples: {len(ds)}")
        print(f"  Crop size: 512×512")

        if len(ds) > 0:
            sample = ds[0]
            print(f"\nSample keys: {list(sample.keys())}")
            print(f"  rgb:   {tuple(sample['rgb'].shape)}")
            print(f"  aolp:  {tuple(sample['aolp'].shape)}")
            print(f"  dolp:  {tuple(sample['dolp'].shape)}")
            print(f"  nir:   {tuple(sample['nir'].shape)}")
            print(f"  label: {tuple(sample['label'].shape)}")

            # Check value ranges
            print(f"\nValue ranges:")
            print(f"  rgb:   [{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
            print(f"  aolp:  [{sample['aolp'].min():.3f}, {sample['aolp'].max():.3f}]")
            print(f"  dolp:  [{sample['dolp'].min():.3f}, {sample['dolp'].max():.3f}]")
            print(f"  nir:   [{sample['nir'].min():.3f}, {sample['nir'].max():.3f}]")
            print(f"  label: [{sample['label'].min()}, {sample['label'].max()}]")

            print("\n✅ Dataset loading: PASSED\n")
        else:
            print("\n⚠️  Dataset is empty, but loading succeeded\n")

    except FileNotFoundError as e:
        print(f"\n⚠️  Dataset not found at {data_root}")
        print(f"   Error: {e}")
        print(f"   Skipping dataset test (this is OK for first run)\n")


def test_training_step():
    """Test a single training step"""
    print("=" * 60)
    print("TEST 3: Training Step (Synthetic Data)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Model
    model = QuadWaterfall(num_classes=20, pretrained=False).to(device)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-6,
        weight_decay=1e-2,
        eps=1e-8,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    # Synthetic batch
    B, H, W = 2, 512, 512
    rgb = torch.randn(B, 3, H, W, device=device)
    aolp = torch.randn(B, 1, H, W, device=device)
    dolp = torch.randn(B, 1, H, W, device=device)
    nir = torch.randn(B, 1, H, W, device=device)
    label = torch.randint(0, 20, (B, H, W), device=device)

    print("Training step:")
    print(f"  Batch size: {B}")
    print(f"  Resolution: {H}×{W}")
    print(f"  Device: {device}\n")

    # Forward
    print("  • Forward pass... ", end='', flush=True)
    output = model(rgb, aolp, dolp, nir)
    print(f"✓ output shape: {tuple(output.shape)}")

    # Loss
    print("  • Loss computation... ", end='', flush=True)
    loss = criterion(output, label)
    print(f"✓ loss: {loss.item():.4f}")

    # Backward
    print("  • Backward pass... ", end='', flush=True)
    optimizer.zero_grad()
    loss.backward()
    print("✓")

    # Optimizer step
    print("  • Optimizer step... ", end='', flush=True)
    optimizer.step()
    print("✓\n")

    print("✅ Training step: PASSED\n")


def test_dataloader():
    """Test dataloader (if dataset exists)"""
    print("=" * 60)
    print("TEST 4: DataLoader")
    print("=" * 60)

    try:
        train_loader, val_loader, test_loader = build_loaders(
            root="../multimodal_dataset",
            crop_size=512,
            batch_size=2,
            num_workers=0,
        )

        print("DataLoaders created successfully")
        print(f"  Train loader: {len(train_loader)} batches")
        print(f"  Val loader:   {len(val_loader)} batches")
        print(f"  Test loader:  {len(test_loader)} batches\n")

        # Try one batch
        print("Fetching one training batch... ", end='', flush=True)
        batch = next(iter(train_loader))
        print("✓")
        print(f"  Keys: {list(batch.keys())}")
        print(f"  Batch shapes:")
        print(f"    rgb:   {tuple(batch['rgb'].shape)}")
        print(f"    aolp:  {tuple(batch['aolp'].shape)}")
        print(f"    dolp:  {tuple(batch['dolp'].shape)}")
        print(f"    nir:   {tuple(batch['nir'].shape)}")
        print(f"    label: {tuple(batch['label'].shape)}\n")

        print("✅ DataLoader: PASSED\n")

    except FileNotFoundError as e:
        print(f"\n⚠️  Dataset not found at /path/to/MCubeS")
        print(f"   Error: {e}")
        print(f"   Skipping dataloader test (this is OK for first run)\n")


def main():
    print("\n" + "=" * 60)
    print("QUADWATERFALL INTEGRATION TEST")
    print("=" * 60 + "\n")

    # Test 1: Model forward pass
    test_model_forward()

    # Test 2: Dataset loading (if available)
    test_dataset_loading()

    # Test 3: Training step
    test_training_step()

    # Test 4: DataLoader
    test_dataloader()

    print("=" * 60)
    print("ALL TESTS COMPLETED! ✅")
    print("=" * 60)
    print("\nYou can now train with:")
    print("  python train.py --batch-size 2 --num-epochs 500 --device cuda\n")
    print("Or with default config (already points to ../multimodal_dataset):")
    print("  python train.py\n")


if __name__ == "__main__":
    main()
