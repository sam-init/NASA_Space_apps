"""
train_cnn.py

Usage:
    python train_cnn.py

Assumptions:
- metadata CSV at OUTPUT_DIR/metadata.csv with columns:
    file,event,points,label  (label optional; 0=no anomaly, 1=anomaly)
- each file is a .npy array of shape (SAMPLES, 2) containing [flux_norm, err_norm]
- adjust paths / hyperparams in CONFIG below
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random

# ---------------- CONFIG ----------------
OUTPUT_DIR = r"D:\exoplanet_ai\processed"
METADATA_CSV = METADATA_CSV = r"D:\exoplanet_ai\processed\metadata_labeled.csv"
SAMPLES = 256
BATCH_SIZE = 4
LR = 1e-4
EPOCHS = 20
WEIGHT_DECAY = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE = os.path.join(OUTPUT_DIR, "cnn_microlens_best.pt")
RANDOM_SEED = 42
# ----------------------------------------

def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything()

# ---------- Dataset ----------
class MicrolensNpyDataset(Dataset):
    def __init__(self, metadata_df, transform=None):
        """
        metadata_df: pandas DataFrame with column 'file' and optionally 'label'
        Each file is a .npy with shape (SAMPLES, 2)
        """
        self.df = metadata_df.reset_index(drop=True)
        if "label" not in self.df.columns:
            print("[WARN] metadata CSV has no 'label' column. All labels set to 0. Edit metadata.csv to add real labels.")
            self.df["label"] = 0
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["file"])  # shape (SAMPLES, 2)
        # ensure shape correctness
        if x.ndim != 2 or x.shape[0] != SAMPLES or x.shape[1] != 2:
            # try transpose or pad/trim
            x = np.asarray(x)
            if x.shape[0] == 2 and x.shape[1] == SAMPLES:
                x = x.T
            else:
                # pad or trim time axis
                x2 = np.zeros((SAMPLES, 2), dtype=np.float32)
                L = min(SAMPLES, x.shape[0])
                x2[:L, :min(2, x.shape[1])] = x[:L, :2] if x.shape[1] >= 2 else np.stack([x[:L,0], np.zeros(L)], axis=1)
                x = x2
        # Convert to float32
        x = x.astype(np.float32)
        y = float(row["label"])
        # return tensor shape (SAMPLES, 2); model will permute to (B, C, L)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ---------- Model ----------
class CNNMicrolens(nn.Module):
    def __init__(self, in_ch=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        # input x shape: (batch, SAMPLES, channels)
        x = x.permute(0, 2, 1)  # -> (batch, channels, length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.global_pool(x).squeeze(-1)  # (batch, 128)
        x = F.relu(self.fc(x))
        x = self.dropout(x)
        return self.out(x).squeeze(-1)  # logits (before sigmoid)


# ---------- Helpers ----------
def collate_fn(batch):
    X = torch.stack([b[0] for b in batch], dim=0)  # (B, SAMPLES, 2)
    y = torch.stack([b[1] for b in batch], dim=0)
    return X, y

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    return {"precision": prec, "recall": recall, "f1": f1, "auc": auc}

# ---------- Main training function ----------
def main():
    # load metadata
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_CSV}")
    meta = pd.read_csv(METADATA_CSV)
    # basic sanity: file paths might be relative; make absolute if needed
    meta["file"] = meta["file"].apply(lambda p: p if os.path.isabs(p) else os.path.join(OUTPUT_DIR, p))

    if "label" not in meta.columns:
        print("[WARN] metadata.csv missing 'label' — training will assume all zeros. Edit file to add true labels for supervised training.")
        meta["label"] = 0

    # split train/val/test (stratify if labels contain both classes)
    if len(meta["label"].unique()) > 1:
        stratify = meta["label"]
    else:
        stratify = None
    train_df, test_df = train_test_split(meta, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_SEED, stratify=(train_df["label"] if stratify is not None else None))

    print(f"[INFO] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = MicrolensNpyDataset(train_df)
    val_ds = MicrolensNpyDataset(val_df)
    test_ds = MicrolensNpyDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = CNNMicrolens().to(DEVICE)
    # compute pos_weight for BCEWithLogitsLoss if imbalance
    labels = train_df["label"].values
    num_pos = labels.sum()
    num_neg = len(labels) - num_pos
    if num_pos == 0:
        pos_weight = torch.tensor(1.0, device=DEVICE)
    else:
        pos_weight = torch.tensor(max(1.0, num_neg / max(1.0, num_pos)), device=DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))
        # --- validation ---
        model.eval()
        val_logits = []
        val_targets = []
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                val_logits.append(logits.detach().cpu().numpy())
                val_targets.append(yb.detach().cpu().numpy())

        val_logits = np.concatenate(val_logits) if val_logits else np.array([])
        val_targets = np.concatenate(val_targets) if val_targets else np.array([])
        # convert logits -> probabilities
        val_probs = 1.0 / (1.0 + np.exp(-val_logits)) if val_logits.size else np.array([])
        metrics = compute_metrics(val_targets, val_probs) if val_probs.size else {"precision":0,"recall":0,"f1":0,"auc":float("nan")}
        avg_val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} val_f1={metrics['f1']:.4f} val_prec={metrics['precision']:.4f} val_rec={metrics['recall']:.4f} val_auc={metrics['auc']:.4f}")

        scheduler.step(avg_val_loss)

        # checkpoint best
        if metrics["f1"] > best_val_f1:
            best_val_f1 = metrics["f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1
            }, MODEL_SAVE)
            print(f"[SAVED] New best model (f1={best_val_f1:.4f}) -> {MODEL_SAVE}")

    # ---------- Final evaluation on test set ----------
    print("\n=== Testing on held-out test set ===")
    model.eval()
    test_logits = []
    test_targets = []
    with torch.no_grad():
        for Xb, yb in tqdm(test_loader, desc="Test"):
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(Xb)
            test_logits.append(logits.detach().cpu().numpy())
            test_targets.append(yb.detach().cpu().numpy())

    test_logits = np.concatenate(test_logits) if test_logits else np.array([])
    test_targets = np.concatenate(test_targets) if test_targets else np.array([])
    test_probs = 1.0 / (1.0 + np.exp(-test_logits)) if test_logits.size else np.array([])
    test_metrics = compute_metrics(test_targets, test_probs) if test_probs.size else {"precision":0,"recall":0,"f1":0,"auc":float("nan")}
    print("Test results:", test_metrics)

    print("\nDone.")

if __name__ == "__main__":
    main()
