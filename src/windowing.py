"""
Windowing and horizon dataset transformation functions for time series forecasting.
"""

from typing import Tuple, Union
import numpy as np
import pandas as pd


def train_test_split_temporal(
    data: Union[pd.DataFrame, np.ndarray],
    train_ratio: float = 0.8
) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
    """
    Splits time series data chronologically (temporal split) to prevent lookahead data leakage.
    """
    split_size = int(len(data) * train_ratio)
    train_data = data[:split_size]
    test_data = data[split_size:]
    return train_data, test_data


def get_labelled_windows(x: np.ndarray, horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a windowed 2D array into input windows and horizon targets.
    """
    return x[:, :-horizon], x[:, -horizon:]


def make_windows(
    x: np.ndarray,
    window_size: int = 7,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turns a 1D or 2D array into sliding windows with target horizons.
    
    Args:
        x: Input time series values (1D or 2D)
        window_size: Number of past time steps to look back
        horizon: Number of future time steps to predict
        
    Returns:
        (windows, labels): Tuple of numpy arrays
    """
    # 1. Create a window with total length = window_size + horizon
    window_step = np.expand_dims(np.arange(window_size + horizon), axis=0)
    
    # 2. Create 2D array of sliding indices
    window_indexes = window_step + np.expand_dims(np.arange(len(x) - (window_size + horizon - 1)), axis=0).T
    
    # 3. Index the original data
    windowed_array = x[window_indexes]
    
    # 4. Get labelled windows
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels


def make_train_test_windows(
    windows: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits windowed dataset into train and test sets chronologically.
    """
    split_size = int(len(windows) * train_ratio)
    X_train = windows[:split_size]
    y_train = labels[:split_size]
    X_test = windows[split_size:]
    y_test = labels[split_size:]
    return X_train, y_train, X_test, y_test
