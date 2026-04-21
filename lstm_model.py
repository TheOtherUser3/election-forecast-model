from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
)

DATA_NPZ = "lstm_data.npz"
TEST_YEAR = 2022
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL
class ElectionLSTM(nn.Module):
    def __init__(self, seq_input_dim, static_dim, hidden_size=128,
                 num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.static_bn = nn.BatchNorm1d(static_dim) if static_dim > 0 else None
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        # final layer's hidden state, (batch, hidden_size)
        h = h[-1]  
        if self.static_bn is not None and static.shape[0] > 1:
            static = self.static_bn(static)
        x = torch.cat([h, static], dim=1)
        return self.head(x).squeeze(-1)


# DATA

@dataclass
class SplitData:
    seq_train: np.ndarray
    static_train: np.ndarray
    y_train: np.ndarray
    seq_test: np.ndarray
    static_test: np.ndarray
    y_test: np.ndarray
    seq_input_dim: int
    static_dim: int


def load_and_split(seq_len_weeks=12):
    """Load npz and produce scaled train/test arrays, truncated to seq_len_weeks."""
    d = np.load(DATA_NPZ, allow_pickle=True)
    sequences = d["sequences"]
    masks = d["masks"]
    X_static = d["X_static"]
    y = d["y"]
    years = d["years"]

    full_weeks = sequences.shape[1]

    # Make sure I didn't mess things up horribly with the bucketing!
    if seq_len_weeks > full_weeks:
        raise ValueError(f"Requested {seq_len_weeks} weeks but data only has {full_weeks}")

    # Keep the most recent seq_len_weeks
    # index 0 = oldest in our data and index -1 = election week
    sequences = sequences[:, -seq_len_weeks:]
    masks = masks[:, -seq_len_weeks:]

    # Stack value + mask into the (N, T, 2) input
    seq_input = np.stack([sequences, masks], axis=-1).astype(np.float32)

    train_idx = years < TEST_YEAR
    test_idx = years == TEST_YEAR

    # Scale static features on train set
    static_scaler = StandardScaler().fit(X_static[train_idx])
    static_train = static_scaler.transform(X_static[train_idx]).astype(np.float32)
    static_test = static_scaler.transform(X_static[test_idx]).astype(np.float32)

    # Scale poll_pct channel on train only; leave mask channel alone
    poll_scaler = StandardScaler().fit(seq_input[train_idx, :, 0].reshape(-1, 1))

    def scale_poll_channel(x):
        x = x.copy()
        n, t, _ = x.shape
        x[:, :, 0] = poll_scaler.transform(x[:, :, 0].reshape(-1, 1)).reshape(n, t)
        return x

    seq_train = scale_poll_channel(seq_input[train_idx])
    seq_test = scale_poll_channel(seq_input[test_idx])

    return SplitData(
        seq_train=seq_train,
        static_train=static_train,
        y_train=y[train_idx].astype(np.float32),
        seq_test=seq_test,
        static_test=static_test,
        y_test=y[test_idx].astype(np.float32),
        seq_input_dim=seq_input.shape[-1],
        static_dim=X_static.shape[1],
    )


# TRAIN + METRICS

def train_and_eval(hidden_size=128, seq_len_weeks=12, epochs=60,
                   batch_size=64, lr=1e-3, weight_decay=1e-5, seed=42,
                   verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = load_and_split(seq_len_weeks=seq_len_weeks)

    X_seq_train = torch.from_numpy(data.seq_train)
    X_static_train = torch.from_numpy(data.static_train)
    y_train = torch.from_numpy(data.y_train)
    X_seq_test = torch.from_numpy(data.seq_test).to(DEVICE)
    X_static_test = torch.from_numpy(data.static_test).to(DEVICE)
    y_test_np = data.y_test

    train_loader = DataLoader(
        TensorDataset(X_seq_train, X_static_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    model = ElectionLSTM(
        seq_input_dim=data.seq_input_dim,
        static_dim=data.static_dim,
        hidden_size=hidden_size,
    ).to(DEVICE)

    # Class imbalance handling: pos_weight = n_neg / n_pos
    # Thank you Claude AI for helping with this part! I was doing it wrong before.
    n_pos = float(data.y_train.sum())
    n_neg = float(len(data.y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0), device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for seq_b, static_b, y_b in train_loader:
            seq_b = seq_b.to(DEVICE)
            static_b = static_b.to(DEVICE)
            y_b = y_b.to(DEVICE)

            optimizer.zero_grad()
            logits = model(seq_b, static_b)
            loss = loss_fn(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running += loss.item() * y_b.size(0)
            n += y_b.size(0)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"epoch {epoch+1:3d}  loss={running/n:.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_seq_test, X_static_test)
        probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs > 0.5).astype(int)

    return {
        "hidden_size": hidden_size,
        "seq_len_weeks": seq_len_weeks,
        "seq_len_days": seq_len_weeks * 7,
        "n_train": int(len(data.y_train)),
        "n_test": int(len(y_test_np)),
        "accuracy": float(accuracy_score(y_test_np, preds)),
        "precision": float(precision_score(y_test_np, preds, zero_division=0)),
        "recall": float(recall_score(y_test_np, preds, zero_division=0)),
        "f1": float(f1_score(y_test_np, preds, zero_division=0)),
        "brier": float(brier_score_loss(y_test_np, probs)),
    }


# Had Claude AI help me write this function to run all the sweeps and print nicely!
def run_all_sweeps():
    print("=" * 70)
    print("Sweep A: Hidden size (seq_len = 12 weeks / 90 days)")
    print("=" * 70)
    hidden_results = []
    for hs in (64, 128, 256):
        print(f"\n--- hidden_size = {hs} ---")
        r = train_and_eval(hidden_size=hs, seq_len_weeks=12, verbose=True)
        print(r)
        hidden_results.append(r)

    print("\n" + "=" * 70)
    print("Sweep B: Sequence length (hidden_size = 128)")
    print("=" * 70)
    seqlen_results = []
    for weeks in (4, 8, 12):
        print(f"\n--- seq_len = {weeks} weeks ({weeks*7} days) ---")
        r = train_and_eval(hidden_size=128, seq_len_weeks=weeks, verbose=True)
        print(r)
        seqlen_results.append(r)

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nHidden size sweep (seq=12w):")
    print(f"{'hidden':>7} | {'acc':>6} | {'prec':>6} | {'recall':>6} | {'f1':>6} | {'brier':>7}")
    for r in hidden_results:
        print(f"{r['hidden_size']:>7} | {r['accuracy']:>6.3f} | {r['precision']:>6.3f} | "
              f"{r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['brier']:>7.4f}")

    print("\nSequence length sweep (hidden=128):")
    print(f"{'weeks':>7} | {'acc':>6} | {'prec':>6} | {'recall':>6} | {'f1':>6} | {'brier':>7}")
    for r in seqlen_results:
        print(f"{r['seq_len_weeks']:>7} | {r['accuracy']:>6.3f} | {r['precision']:>6.3f} | "
              f"{r['recall']:>6.3f} | {r['f1']:>6.3f} | {r['brier']:>7.4f}")

    return hidden_results, seqlen_results


if __name__ == "__main__":
    run_all_sweeps()

""" FILE DESCRIPTION
LSTM model for predicting U.S. House and Senate election outcomes.

Reads lstm_data.npz (from build_lstm_features.py), trains an LSTM on a sliding
window of pre-election weekly poll averages plus static features (fundraising,
incumbency, partisanship), evaluates on 2022, and runs two
sweeps:

  Sweep A - hidden size {64, 128, 256}
  Sweep B - sequence length in days {30, 60, 90} 

Also prints accuracy, precision, recall, f1, and brier scores for each sweep.

Since we hadn't learned this in lecture, I received a lot of help from Claude AI to implement the LSTM and the training loop,
especially in terms of learning how to implement such a model in PyTorch and how to handle class imbalance.
"""
