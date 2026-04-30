"""
Bayesian Hyperparameter Search for QuadWaterfall Semantic Segmentation
Uses Optuna to find optimal hyperparameters for the hybrid FuseForm + WTPose architecture
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm
import torch

from train import Trainer
from config import Config, config as default_config
from architecture import QuadWaterfall


class HyperparameterObjective:
    """Optuna objective function that wraps the Trainer for a single trial"""

    def __init__(
        self,
        epochs_per_trial,
        batch_size,
        data_root,
        device,
        dropout_range=(0.05, 0.3),
        poly_power_range=(0.7, 1.0),
    ):
        self.epochs_per_trial = epochs_per_trial
        self.batch_size = batch_size
        self.data_root = data_root
        self.device = device
        self.dropout_range = dropout_range
        self.poly_power_range = poly_power_range

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Define hyperparameter search space for semantic segmentation
        Returns best validation mIoU achieved during this trial
        """

        # ── Hyperparameter search space ──────────────────────────────────
        # Learning rate (polynomial schedule parameters)
        # Paper uses: lr_max=6e-6, lr_init=6e-7, lr_min=1e-9
        # Search ±2-3x around paper values for small dataset stability
        lr_max = trial.suggest_float("lr_max", 2e-6, 1.5e-5, log=True)    # 6e-6 ± 3x/2.5x
        lr_init = trial.suggest_float("lr_init", 2e-7, 1.5e-6, log=True)  # 6e-7 ± 3x/2.5x (balanced)
        lr_min = trial.suggest_float("lr_min", 1e-10, 1e-8, log=True)     # 1e-9 ± 10x/10x

        # Optimizer regularization
        # Paper uses: weight_decay=1e-2
        weight_decay = trial.suggest_float("weight_decay", 5e-3, 3e-2, log=True)  # 1e-2 ± 2x/3x (balanced)

        # Scheduler parameters
        warmup_epochs = trial.suggest_int("warmup_epochs", 2, 15)
        poly_power = trial.suggest_float(
            "poly_power", self.poly_power_range[0], self.poly_power_range[1]
        )

        # Dropout rate for regularization
        dropout_p = trial.suggest_float(
            "dropout_p", self.dropout_range[0], self.dropout_range[1]
        )

        # ── Log selected hyperparameters ─────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  🔍 Trial {trial.number} — Hyperparameters")
        print(f"{'='*70}")
        print(f"  lr_max:         {lr_max:.6e}")
        print(f"  lr_init:        {lr_init:.6e}")
        print(f"  lr_min:         {lr_min:.6e}")
        print(f"  weight_decay:   {weight_decay:.6e}")
        print(f"  warmup_epochs:  {warmup_epochs}")
        print(f"  poly_power:     {poly_power:.4f}")
        print(f"  dropout_p:      {dropout_p:.4f}")
        print(f"{'='*70}\n")
        sys.stdout.flush()

        # Fixed parameters (proven from papers, not worth searching)
        optimizer = "adamw"
        adam_epsilon = 1e-8
        loss_fn = "cross_entropy"

        # ── Create config object ─────────────────────────────────────────
        cfg = Config()
        cfg.data.root = self.data_root

        cfg.train.batch_size = self.batch_size
        cfg.train.num_epochs = self.epochs_per_trial
        cfg.train.lr_init = lr_init
        cfg.train.lr_max = lr_max
        cfg.train.lr_min = lr_min
        cfg.train.warmup_epochs = warmup_epochs
        cfg.train.weight_decay = weight_decay
        cfg.train.poly_power = poly_power
        cfg.train.adam_epsilon = adam_epsilon
        cfg.train.device = self.device

        # ── Create model with dropout ────────────────────────────────────
        model = QuadWaterfall(
            num_classes=cfg.data.num_classes,
            rgb_var="b4",
            aux_var="b2",
            pretrained=True,
            p=dropout_p,  # Variable dropout from trial
            enc_checkpoint=False,
            qwtm_checkpoint=False,
        )

        # ── Create trainer ───────────────────────────────────────────────
        trainer = Trainer(cfg, model, device=self.device)

        # ── Training loop with Optuna pruning ────────────────────────────
        val_miou_history = []

        try:
            with tqdm(total=cfg.train.num_epochs, desc=f"Trial {trial.number}",
                     leave=True, position=0, ncols=100) as pbar:
                for epoch in range(cfg.train.num_epochs):
                    try:
                        # Train for one epoch
                        train_loss, train_metrics = trainer.train_epoch(epoch)

                        pbar.update(1)
                        sys.stdout.flush()

                        # Validate
                        if (epoch + 1) % cfg.eval.val_interval == 0:
                            val_loss, val_metrics = trainer.validate(epoch)
                            val_miou = val_metrics["mIoU"]
                            val_miou_history.append(val_miou)

                            # Update learning rate scheduler
                            trainer.scheduler.step(epoch)

                            # Report to Optuna for pruning
                            trial.report(val_miou, epoch)

                            # Print detailed progress
                            print(f"\n  Trial {trial.number} | Epoch {epoch + 1}/{cfg.train.num_epochs} | "
                                  f"Val mIoU: {val_miou:.4f} | LR: {trainer.scheduler.get_lr():.2e}")
                            sys.stdout.flush()

                            pbar.set_postfix({"mIoU": f"{val_miou:.4f}"})

                            # Check if trial should be pruned
                            if trial.should_prune():
                                print(f"\n  ⚠️  Trial {trial.number} PRUNED at epoch {epoch + 1} "
                                      f"(mIoU={val_miou:.4f})")
                                sys.stdout.flush()
                                raise optuna.TrialPruned()

                    except optuna.TrialPruned:
                        raise
                    except Exception as epoch_error:
                        print(f"\n  ❌ Trial {trial.number} EPOCH ERROR at epoch {epoch + 1}: {str(epoch_error)}")
                        print(f"     Type: {type(epoch_error).__name__}")
                        import traceback
                        traceback.print_exc()
                        sys.stdout.flush()
                        raise

            # Return the best mIoU achieved during this trial
            best_miou = max(val_miou_history) if val_miou_history else 0.0
            print(f"\n  ✅ Trial {trial.number} COMPLETE | Best mIoU: {best_miou:.4f}\n")
            sys.stdout.flush()
            return best_miou

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"\n  ❌ Trial {trial.number} FAILED | Error: {str(e)}")
            print(f"     Type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            raise

        finally:
            # Clean up GPU memory even if trial is pruned or fails
            del trainer
            del model
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_search(args):
    """Run the Optuna hyperparameter search with the defined objective function"""

    # ── Output directory ─────────────────────────────────────────────────
    results_dir = Path("optuna_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    db_path = results_dir / "quadwaterfall_study.db"

    print(f"\n{'='*70}")
    print(f"  Bayesian Hyperparameter Search — QuadWaterfall Semantic Segmentation")
    print(f"{'='*70}")
    print(f"  Trials:           {args.n_trials}")
    print(f"  Epochs per trial: {args.epochs_per_trial}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Device:           {args.device}")
    print(f"  Data root:        {args.data_root}")
    print(f"  Results dir:      {results_dir}")
    print(f"  Database:         {db_path}")
    print(f"  Metric:           mIoU (maximize)")
    print(f"{'='*70}\n")

    # ── Create Optuna study ──────────────────────────────────────────────
    study = optuna.create_study(
        study_name="quadwaterfall_hyperparameter_search",
        direction="maximize",  # Maximize validation mIoU
        sampler=TPESampler(n_startup_trials=3, seed=42),  # 3 random, then Bayesian
        pruner=MedianPruner(
            n_startup_trials=3,  # Start pruning after 3 trials
            n_warmup_steps=2,  # Start pruning after epoch 2
        ),
        storage=f"sqlite:///{db_path}",  # SQLite database for persistence
        load_if_exists=True,  # Resume from previous run
    )

    # ── Run optimization ─────────────────────────────────────────────────
    objective = HyperparameterObjective(
        epochs_per_trial=args.epochs_per_trial,
        batch_size=args.batch_size,
        data_root=args.data_root,
        device=args.device,
        dropout_range=(args.dropout_min, args.dropout_max),
        poly_power_range=(args.poly_power_min, args.poly_power_max),
    )

    start_time = time.time()
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n⚠️  Search interrupted by user. Saving current results...")

    elapsed = time.time() - start_time

    # ── Report results ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Search Complete!")
    print(f"{'='*70}")
    print(f"  Total time:        {elapsed / 3600:.2f} hours")
    print(f"  Trials completed:  {len(study.trials)}")
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"  Trials pruned:     {pruned_count}")

    if len(study.trials) == 0:
        print("\n❌ No trials completed. Exiting.")
        return

    try:
        best_miou = study.best_value
        best_trial_num = study.best_trial.number
        best_params = study.best_params
    except ValueError:
        print("\n❌ No successful trials. Cannot extract best parameters.")
        return

    print(f"  Best mIoU:         {best_miou:.4f}")
    print(f"  Best trial:        {best_trial_num}")
    print(f"\n  Best hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"    {key:20s}: {value:.6f}")
        else:
            print(f"    {key:20s}: {value}")

    # ── Save best params to JSON ─────────────────────────────────────────
    best_params_dict = {
        "version": "quadwaterfall_hybrid",
        "best_miou": float(best_miou),
        "best_trial": best_trial_num,
        "n_trials": len(study.trials),
        "n_pruned": pruned_count,
        "epochs_per_trial": args.epochs_per_trial,
        "batch_size": args.batch_size,
        "elapsed_hours": elapsed / 3600,
        "timestamp": datetime.now().isoformat(),
        "params": best_params,
    }

    best_file = results_dir / "best_params.json"
    with open(best_file, "w") as f:
        json.dump(best_params_dict, f, indent=2)
    print(f"\n  ✓ Best params saved to: {best_file}")

    # ── Save all trials to JSON ──────────────────────────────────────────
    all_trials = []
    for t in study.trials:
        trial_data = {
            "number": t.number,
            "state": t.state.name,
            "value": float(t.value) if t.value is not None else None,
            "params": t.params,
            "duration_seconds": (t.datetime_complete - t.datetime_start).total_seconds()
            if t.datetime_complete and t.datetime_start
            else None,
        }
        all_trials.append(trial_data)

    trials_file = results_dir / "all_trials.json"
    with open(trials_file, "w") as f:
        json.dump(all_trials, f, indent=2)
    print(f"  ✓ All trials saved to: {trials_file}")

    # ── Build training command for best params ───────────────────────────
    if not args.skip_full_train:
        print(f"\n{'='*70}")
        print(f"  Launching full training with best hyperparameters")
        print(f"     Epochs: {args.full_epochs}")
        print(f"{'='*70}\n")

        _train_best(
            best_params,
            args.full_epochs,
            args.batch_size,
            args.data_root,
            args.device,
        )
    else:
        cmd = _build_train_command(best_params, args.full_epochs, args.batch_size)
        print(f"\n  Skipping full training (--skip-full-train).")
        print(f"  To train manually with best params, create train.py args or modify config.py:")
        print(f"\n  Key parameters to use:")
        for key, value in best_params.items():
            if isinstance(value, float):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        print(f"\n{'='*70}\n")


def _build_train_command(bp, full_epochs, batch_size):
    """Build a command line string showing how to use the best params"""
    cmd = "# Best hyperparameters for config.py TrainConfig:\n"
    cmd += f"lr_init = {bp['lr_init']:.6e}\n"
    cmd += f"lr_max = {bp['lr_max']:.6e}\n"
    cmd += f"lr_min = {bp['lr_min']:.6e}\n"
    cmd += f"weight_decay = {bp['weight_decay']:.6e}\n"
    cmd += f"warmup_epochs = {bp['warmup_epochs']}\n"
    cmd += f"poly_power = {bp['poly_power']:.6f}\n"
    cmd += f"# Model dropout_p = {bp['dropout_p']:.6f}\n"
    cmd += f"# Batch size = {batch_size}, num_epochs = {full_epochs}\n"
    return cmd


def _train_best(best_params, full_epochs, batch_size, data_root, device):
    """
    Train the model with the best hyperparameters found by Optuna.
    This runs a full training session with the best params for full_epochs.
    """
    print("Creating model and trainer with best hyperparameters...\n")

    # ── Create config with best params ───────────────────────────────────
    cfg = Config()
    cfg.data.root = data_root

    cfg.train.batch_size = batch_size
    cfg.train.num_epochs = full_epochs
    cfg.train.lr_init = best_params["lr_init"]
    cfg.train.lr_max = best_params["lr_max"]
    cfg.train.lr_min = best_params["lr_min"]
    cfg.train.weight_decay = best_params["weight_decay"]
    cfg.train.warmup_epochs = best_params["warmup_epochs"]
    cfg.train.poly_power = best_params["poly_power"]
    cfg.train.device = device

    # ── Create model with best dropout ───────────────────────────────────
    model = QuadWaterfall(
        num_classes=cfg.data.num_classes,
        rgb_var="b4",
        aux_var="b2",
        pretrained=True,
        p=best_params["dropout_p"],
        enc_checkpoint=True,
        qwtm_checkpoint=True,
    )

    # ── Create trainer and train ─────────────────────────────────────────
    trainer = Trainer(cfg, model, device=device)
    trainer.train()

    print(f"\n{'='*70}")
    print(f"  ✅ Full training complete!")
    print(f"  Best val mIoU: {trainer.best_miou:.4f}")
    print(f"  Checkpoint dir: {trainer.checkpoint_dir}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian Hyperparameter Search for QuadWaterfall",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search with 20 trials, 10 epochs per trial, then auto-train for 100 epochs
  python hyperparameter_search.py --n-trials 20 --epochs-per-trial 10 --full-epochs 100

  # Search only (don't auto-train)
  python hyperparameter_search.py --n-trials 20 --epochs-per-trial 10 --skip-full-train

  # Resume a previous search (same study name, adds more trials)
  python hyperparameter_search.py --n-trials 10 --epochs-per-trial 10
        """,
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=15,
        help="Number of Optuna trials to run (default: 15)",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=12,
        help="Training epochs per trial (default: 12). Shorter = faster search, but less reliable.",
    )
    parser.add_argument(
        "--full-epochs",
        type=int,
        default=100,
        help="Epochs for full training with best params (default: 100)",
    )
    parser.add_argument(
        "--skip-full-train",
        action="store_true",
        help="Only search, don't auto-train best params",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2, limited by GPU memory)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../multimodal_dataset",
        help="Path to MCubeS dataset (default: ../multimodal_dataset)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device selection (default: cuda)",
    )
    parser.add_argument(
        "--dropout-min",
        type=float,
        default=0.05,
        help="Minimum dropout rate to search (default: 0.05)",
    )
    parser.add_argument(
        "--dropout-max",
        type=float,
        default=0.3,
        help="Maximum dropout rate to search (default: 0.3)",
    )
    parser.add_argument(
        "--poly-power-min",
        type=float,
        default=0.7,
        help="Minimum polynomial decay power (default: 0.7)",
    )
    parser.add_argument(
        "--poly-power-max",
        type=float,
        default=1.0,
        help="Maximum polynomial decay power (default: 1.0)",
    )

    args = parser.parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
