"""
Training pipeline for Bitcoin Time Series Forecasting using TensorFlow / Keras.
Supports training Dense, Conv1D, LSTM, and N-BEATS models on windowed Bitcoin data.
"""

import argparse
import os
import numpy as np

from src.data_loader import load_bitcoin_data
from src.windowing import make_windows, make_train_test_windows
from src.metrics import evaluate_preds


def train_model(
    model_type: str = "dense",
    window_size: int = 7,
    horizon: int = 1,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    save_path: str = "checkpoints/best_model.keras"
):
    print("=" * 60)
    print(f"TensorFlow Training: {model_type.upper()} Model (Window={window_size}, Horizon={horizon})")
    print("=" * 60)

    # 1. Load Bitcoin Data
    prices_df = load_bitcoin_data()
    prices = prices_df["Price"].to_numpy()
    print(f"Loaded {len(prices)} days of Bitcoin history ({prices_df.index[0].date()} to {prices_df.index[-1].date()})")

    # 2. Windowing
    windows, labels = make_windows(prices, window_size=window_size, horizon=horizon)
    X_train, y_train, X_test, y_test = make_train_test_windows(windows, labels, train_ratio=0.8)
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    # 3. Naive Baseline (No deep learning required)
    if model_type.lower() == "naive":
        from src.models_tf import NaiveModel
        naive = NaiveModel(horizon=horizon)
        preds = naive.predict(X_test)
        metrics = evaluate_preds(y_test, preds)
        print("\n--- Naive Baseline Performance ---")
        for k, v in metrics.items():
            print(f"  {k.upper():6s}: {v}")
        return metrics

    # 4. TensorFlow Model Setup
    try:
        import tensorflow as tf
        from src.models_tf import build_dense_model, build_conv1d_model, build_lstm_model, build_nbeats_model
    except ImportError as e:
        print(f"\n[Error] TensorFlow is required for deep learning model training: {e}")
        print("Please install TensorFlow: pip install tensorflow")
        return None

    tf.random.set_seed(42)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    if model_type.lower() == "dense":
        model = build_dense_model(window_size=window_size, horizon=horizon, hidden_units=128)
    elif model_type.lower() == "conv1d":
        model = build_conv1d_model(window_size=window_size, horizon=horizon, filters=128, kernel_size=5)
    elif model_type.lower() == "lstm":
        model = build_lstm_model(window_size=window_size, horizon=horizon, units=128)
    elif model_type.lower() == "nbeats":
        model = build_nbeats_model(input_size=window_size, horizon=horizon, n_neurons=128, n_layers=4, n_stacks=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.compile(
        loss="mae",
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=["mae", "mse"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=30,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=15,
            factor=0.2,
            verbose=1
        )
    ]

    # 5. Fit Model
    print(f"\nTraining {model.name} for up to {epochs} epochs...")
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )

    # 6. Evaluation
    print("\nEvaluating best saved model on test set...")
    if os.path.exists(save_path):
        model = tf.keras.models.load_model(save_path)

    test_preds = model.predict(X_test)
    metrics = evaluate_preds(y_test, test_preds)

    print("\n" + "=" * 40)
    print(f"Final Test Evaluation ({model_type.upper()})")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k.upper():6s}: {v}")
    print("=" * 40)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Bitcoin Time Series Models with TensorFlow")
    parser.add_argument("--model", type=str, default="dense", choices=["naive", "dense", "conv1d", "lstm", "nbeats"])
    parser.add_argument("--window", type=int, default=7, help="Window size (lookback days)")
    parser.add_argument("--horizon", type=int, default=1, help="Horizon size (forecast days)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
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
