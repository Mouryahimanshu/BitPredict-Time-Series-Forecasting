"""
PyTorch implementations of Time Series Forecasting architectures:
  - Naive Baseline
  - Dense Forecaster
  - 1D Convolutional Neural Network (Conv1D)
  - Recurrent Neural Network (LSTM)
  - N-BEATS (Neural Basis Expansion Analysis for Time Series)
  - Ensemble Model with Uncertainty Estimation
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np


class NaiveModel:
    """
    Model 0: Naive forecast baseline.
    Predicts the value from the previous time step: y_t = y_{t-1}.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Takes input window (shape: [N, window_size]) and returns the last known value
        repeated across the horizon.
        """
        x = np.asarray(x)
        if x.ndim == 1:
            return np.repeat(x[-1], self.horizon)
        # Return the last time step of each window
        last_val = x[:, -1:]
        return np.repeat(last_val, self.horizon, axis=1)


class DenseForecaster(nn.Module):
    """
    Models 1, 2, 3, 6: Fully connected multi-layer perceptron.
    """
    def __init__(self, input_dim: int, horizon: int = 1, hidden_units: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has channel/feature dim, flatten
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)


class Conv1DForecaster(nn.Module):
    """
    Model 4: 1D Temporal Convolutional network with causal padding.
    """
    def __init__(self, window_size: int, horizon: int = 1, in_channels: int = 1, num_filters: int = 128, kernel_size: int = 5):
        super().__init__()
        # Causal padding ensures filter doesn't peek into the future
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size - 1
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batch_size, window_size] or [batch_size, window_size, channels]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, window_size]
        elif x.dim() == 3 and x.shape[1] != 1 and x.shape[2] == 1:
            x = x.permute(0, 2, 1)  # [batch_size, 1, window_size]
            
        out = self.relu(self.conv1(x))
        # Trim causal padding overflow
        out = self.pool(out).squeeze(-1)
        return self.fc(out)


class LSTMForecaster(nn.Module):
    """
    Model 5: Long Short-Term Memory (LSTM) Recurrent Neural Network.
    """
    def __init__(self, window_size: int, horizon: int = 1, in_channels: int = 1, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, seq_len, in_channels]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, (hn, _) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)


class NBeatsBlock(nn.Module):
    """
    Custom N-BEATS residual block layer (Oreshkin et al., 2019).
    Computes backcast (reconstruction of past input) and forecast (future prediction)
    using shared dense layers followed by theta projection layers.
    """
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int = 128, n_layers: int = 4):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon

        layers = []
        for i in range(n_layers):
            in_dim = input_size if i == 0 else n_neurons
            layers.append(nn.Linear(in_dim, n_neurons))
            layers.append(nn.ReLU())
        self.fc_stack = nn.Sequential(*layers)

        # Theta linear projections
        self.theta_backcast = nn.Linear(n_neurons, theta_size)
        self.theta_forecast = nn.Linear(n_neurons, theta_size)

        # Basis expansion projections (generic linear basis)
        self.backcast_basis = nn.Linear(theta_size, input_size)
        self.forecast_basis = nn.Linear(theta_size, horizon)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.fc_stack(x)
        theta_b = self.theta_backcast(features)
        theta_f = self.theta_forecast(features)

        backcast = self.backcast_basis(theta_b)
        forecast = self.forecast_basis(theta_f)
        return backcast, forecast


class NBeatsModel(nn.Module):
    """
    Model 7/8: N-BEATS architecture combining stacks of residual blocks.
    Employs doubly residual stacking:
      - Backcast residuals subtract from input signal at each block
      - Forecast signals accumulate additively to form global forecast
    """
    def __init__(self, input_size: int, horizon: int = 1, n_neurons: int = 128, n_layers: int = 4, n_blocks: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(
                input_size=input_size,
                theta_size=input_size + horizon,
                horizon=horizon,
                n_neurons=n_neurons,
                n_layers=n_layers
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuals = x
        forecast_total = 0.0
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast_total = forecast_total + block_forecast
        return forecast_total


class EnsembleForecaster:
    """
    Model 8/9: Deep Ensembling with Uncertainty Estimation.
    Combines predictions from multiple diverse models to produce:
      - Point forecast (ensemble mean)
      - Uncertainty intervals (2.5th and 97.5th percentiles / 95% confidence interval)
    """
    def __init__(self, models: List[nn.Module]):
        self.models = models

    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        preds_list = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                pred = m(x).cpu().numpy()
                preds_list.append(pred)
                
        preds_arr = np.array(preds_list)  # [num_models, batch_size, horizon]
        mean_preds = np.mean(preds_arr, axis=0)
        median_preds = np.median(preds_arr, axis=0)
        lower_bound = np.percentile(preds_arr, 2.5, axis=0)
        upper_bound = np.percentile(preds_arr, 97.5, axis=0)
        return mean_preds, median_preds, lower_bound, upper_bound
