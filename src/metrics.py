"""
Evaluation metrics module for time series forecasting.
Calculates MAE, MSE, RMSE, MAPE, and MASE.
"""

from typing import Dict
import numpy as np


def mean_absolute_scaled_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes Mean Absolute Scaled Error (MASE) against a 1-step naive forecast benchmark.
    MASE < 1 means the model outperforms a naive baseline.
    MASE > 1 means the model performs worse than a naive baseline.
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Naive forecast: y_t = y_{t-1}
    # Scale metric by difference between successive true values
    naive_mae = np.mean(np.abs(y_true[1:] - y_true[:-1]))
    if naive_mae == 0:
        return float("inf")
    return float(mae / naive_mae)


def evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates comprehensive time series forecasting performance metrics.
    
    Returns:
        Dictionary with:
          - mae: Mean Absolute Error
          - mse: Mean Squared Error
          - rmse: Root Mean Squared Error
          - mape: Mean Absolute Percentage Error (%)
          - mase: Mean Absolute Scaled Error
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Flatten if needed
    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.flatten()
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.flatten()
        
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mse = float(np.mean(np.square(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    
    # MAPE calculation handling zero division safely
    denom = np.where(y_true == 0, 1e-8, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    
    mase = mean_absolute_scaled_error(y_true, y_pred)
    
    return {
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
        "mase": round(mase, 4)
    }
