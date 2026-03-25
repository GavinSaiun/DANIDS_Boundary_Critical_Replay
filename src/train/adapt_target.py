from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLP


# =========================
# Config
# =========================
RANDOM_STATE = 42
BATCH_SIZE = 1024
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 12
PATIENCE = 4

LABEL_COL = "Label"
ATTACK_COL = "Attack"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_split(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in [LABEL_COL, ATTACK_COL]]


def load_standardizer(json_path: Path) -> Tuple[pd.Series, pd.Series]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    mean = pd.Series(payload["mean"], dtype="float64")
    std = pd.Series(payload["std"], dtype="float64")
    std = std.replace(0, 1.0)

    return mean, std


def apply_standardizer(df: pd.DataFrame, feature_cols: list[str], mean: pd.Series, std: pd.Series) -> pd.DataFrame:
    df = df.copy()

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(mean)
    df[feature_cols] = (df[feature_cols] - mean) / std
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df


def df_to_tensors(df: pd.DataFrame, feature_cols: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[LABEL_COL].values, dtype=torch.float32)
    return x, y


def make_loader(df: pd.DataFrame, feature_cols: list[str], batch_size: int, shuffle: bool) -> DataLoader:
    x, y = df_to_tensors(df, feature_cols)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits)
        preds = (logits > 0).long()

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_prob)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    y_prob = y_prob[valid_mask]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    total_examples = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_size = xb.size(0)
        running_loss += loss.item() * batch_size
        total_examples += batch_size

    return running_loss / total_examples


def main() -> None:
    set_seed()
    device = get_device()
    repo_root = get_repo_root()

    splits_dir = repo_root / "data" / "splits"
    models_dir = repo_root / "results" / "models"
    logs_dir = repo_root / "results" / "logs"
    ensure_dir(logs_dir)

    print(f"Using device: {device}")

    # =========================
    # Load splits
    # =========================
    source_test = load_split(splits_dir / "source_test.csv")
    target_adapt_train = load_split(splits_dir / "target_adapt_train.csv")
    target_adapt_val = load_split(splits_dir / "target_adapt_val.csv")
    target_test = load_split(splits_dir / "target_test.csv")

    feature_cols = get_feature_cols(target_adapt_train)
    input_dim = len(feature_cols)

    print(f"Number of features: {input_dim}")

    # =========================
    # Load source standardizer
    # =========================
    mean, std = load_standardizer(logs_dir / "source_standardizer.json")

    source_test = apply_standardizer(source_test, feature_cols, mean, std)
    target_adapt_train = apply_standardizer(target_adapt_train, feature_cols, mean, std)
    target_adapt_val = apply_standardizer(target_adapt_val, feature_cols, mean, std)
    target_test = apply_standardizer(target_test, feature_cols, mean, std)

    # =========================
    # Dataloaders
    # =========================
    adapt_train_loader = make_loader(target_adapt_train, feature_cols, BATCH_SIZE, shuffle=True)
    adapt_val_loader = make_loader(target_adapt_val, feature_cols, BATCH_SIZE, shuffle=False)
    target_test_loader = make_loader(target_test, feature_cols, BATCH_SIZE, shuffle=False)
    source_test_loader = make_loader(source_test, feature_cols, BATCH_SIZE, shuffle=False)

    # =========================
    # Load best source model
    # =========================
    model = MLP(input_dim=input_dim).to(device)
    source_model_path = models_dir / "source_mlp_best.pt"
    model.load_state_dict(torch.load(source_model_path, map_location=device))

    print("\nBefore adaptation:")
    pre_source_metrics = evaluate(model, source_test_loader, device)
    pre_target_metrics = evaluate(model, target_test_loader, device)

    print("Source test metrics:")
    for k, v in pre_source_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Target test metrics (zero-shot):")
    for k, v in pre_target_metrics.items():
        print(f"  {k}: {v:.4f}")

    # =========================
    # Fine-tune on target adaptation set
    # =========================
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_model_path = models_dir / "target_adapt_no_replay_best.pt"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, adapt_train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, adapt_val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"target_val_f1={val_metrics['f1']:.4f} | "
            f"target_val_acc={val_metrics['accuracy']:.4f} | "
            f"target_val_recall={val_metrics['recall']:.4f} | "
            f"target_val_auc={val_metrics['roc_auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    print(f"\nBest adaptation epoch: {best_epoch}")
    print(f"Best target val F1: {best_val_f1:.4f}")

    # =========================
    # Load best adapted model
    # =========================
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # =========================
    # Final evaluation
    # =========================
    post_target_metrics = evaluate(model, target_test_loader, device)
    post_source_metrics = evaluate(model, source_test_loader, device)

    print("\nAfter adaptation:")
    print("Target test metrics:")
    for k, v in post_target_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Source test metrics:")
    for k, v in post_source_metrics.items():
        print(f"  {k}: {v:.4f}")

    # =========================
    # Save logs
    # =========================
    history_df = pd.DataFrame(history)
    history_df.to_csv(logs_dir / "target_adaptation_no_replay_history.csv", index=False)

    metrics_df = pd.DataFrame(
        [
            {"phase": "pre_adapt_source_test", **pre_source_metrics},
            {"phase": "pre_adapt_target_test_zero_shot", **pre_target_metrics},
            {"phase": "post_adapt_target_test", **post_target_metrics},
            {"phase": "post_adapt_source_test", **post_source_metrics},
        ]
    )
    metrics_df.to_csv(logs_dir / "target_adaptation_no_replay_metrics.csv", index=False)

    print("\nSaved:")
    print(f"  adapted model: {best_model_path}")
    print(f"  history: {logs_dir / 'target_adaptation_no_replay_history.csv'}")
    print(f"  metrics: {logs_dir / 'target_adaptation_no_replay_metrics.csv'}")


if __name__ == "__main__":
    main()