"""
Training pipeline for Bitcoin Time Series Forecasting using PyTorch.
Supports training Dense, Conv1D, LSTM, and N-BEATS models on windowed Bitcoin data.
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.data_loader import load_bitcoin_data
from src.windowing import make_windows, make_train_test_windows
from src.metrics import evaluate_preds
from src.models_torch import NaiveModel, DenseForecaster, Conv1DForecaster, LSTMForecaster, NBeatsModel


def train_model(
    model_type: str = "dense",
    window_size: int = 7,
    horizon: int = 1,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_path: str = "checkpoints/best_model.pt"
):
    print("=" * 60)
    print(f"Training {model_type.upper()} Model (Window={window_size}, Horizon={horizon})")
    print("=" * 60)

    # 1. Load Data
    prices_df = load_bitcoin_data()
    prices = prices_df["Price"].to_numpy()
    print(f"Loaded {len(prices)} days of Bitcoin price history ({prices_df.index[0].date()} to {prices_df.index[-1].date()})")

    # 2. Windowing
    windows, labels = make_windows(prices, window_size=window_size, horizon=horizon)
    X_train, y_train, X_test, y_test = make_train_test_windows(windows, labels, train_ratio=0.8)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Naive Baseline Check
    if model_type.lower() == "naive":
        naive = NaiveModel(horizon=horizon)
        preds = naive.predict(X_test)
        metrics = evaluate_preds(y_test, preds)
        print("\n--- Naive Baseline Performance ---")
        for k, v in metrics.items():
            print(f"  {k.upper()}: {v}")
        return metrics

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    if model_type.lower() == "dense":
        model = DenseForecaster(input_dim=window_size, horizon=horizon)
    elif model_type.lower() == "conv1d":
        model = Conv1DForecaster(window_size=window_size, horizon=horizon)
    elif model_type.lower() == "lstm":
        model = LSTMForecaster(window_size=window_size, horizon=horizon)
    elif model_type.lower() == "nbeats":
        model = NBeatsModel(input_size=window_size, horizon=horizon)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    criterion = nn.L1Loss()  # MAE Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    best_val_mae = float("inf")

    # 5. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(X_train)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
        val_loss /= len(X_test)

        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs:3d} | Train MAE: {train_loss:8.2f} | Val MAE: {val_loss:8.2f}")

    # 6. Evaluation with best model
    model.load_state_dict(torch.load(save_path))
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_t).cpu().numpy()

    metrics = evaluate_preds(y_test, test_preds)
    print("\n" + "=" * 40)
    print(f"Final Test Evaluation ({model_type.upper()})")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k.upper():6s}: {v}")
    print("=" * 40)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bitcoin Forecasting Models")
    parser.add_argument("--model", type=str, default="dense", choices=["naive", "dense", "conv1d", "lstm", "nbeats"])
    parser.add_argument("--window", type=int, default=7, help="Window size (lookback days)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon size (forecast days)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train_model(
        model_type=args.model,
        window_size=args.window,
        horizon=args.horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
