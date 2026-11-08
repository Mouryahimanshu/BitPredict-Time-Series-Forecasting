"""
TensorFlow 2 / Keras implementations of all Time Series Forecasting architectures:
  - Model 0: Naive Baseline
  - Models 1, 2, 3: Dense Forecaster (Sequential MLP)
  - Model 4: 1D Temporal Convolutional Network (Conv1D with causal padding)
  - Model 5: Recurrent Neural Network (LSTM)
  - Model 6: Multivariate Dense Forecaster (Price + Block Reward Halvings)
  - Model 7: N-BEATS (Neural Basis Expansion Analysis with Doubly Residual Stacking)
  - Model 8: Deep Ensemble with Loss Diversity (MAE, MSE, MAPE) & Uncertainty Estimates
"""

from typing import List, Tuple, Any
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    _BaseLayer = layers.Layer
except ImportError:
    tf = None
    layers = None
    TF_AVAILABLE = False
    _BaseLayer = object


def _check_tf():
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed in the active environment. "
            "To train or evaluate deep learning models, install it via: pip install tensorflow"
        )


class NaiveModel:
    """
    Model 0: Naive Forecast Baseline.
    Predicts the value from the previous time step: y_t = y_{t-1}.
    """
    def __init__(self, horizon: int = 1):
        self.horizon = horizon

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            return np.repeat(x[-1], self.horizon)
        last_val = x[:, -1:]
        return np.repeat(last_val, self.horizon, axis=1)


def build_dense_model(window_size: int, horizon: int = 1, hidden_units: int = 128, name: str = "model_dense"):
    """
    Models 1, 2, 3: Feed-forward Dense Neural Network.
    """
    _check_tf()
    model = tf.keras.Sequential([
        layers.Input(shape=(window_size,)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dense(horizon, activation="linear")
    ], name=name)
    return model


def build_conv1d_model(window_size: int, horizon: int = 1, filters: int = 128, kernel_size: int = 5, name: str = "model_conv1d"):
    """
    Model 4: 1D Temporal Convolutional network with causal padding.
    Causal padding prevents lookahead data leakage into future tokens.
    """
    _check_tf()
    inputs = layers.Input(shape=(window_size,))
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs)
    x = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="causal", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(horizon)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_lstm_model(window_size: int, horizon: int = 1, units: int = 128, name: str = "model_lstm"):
    """
    Model 5: Long Short-Term Memory (LSTM) Recurrent Neural Network.
    """
    _check_tf()
    inputs = layers.Input(shape=(window_size,))
    x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
    x = layers.LSTM(units, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_multivariate_dense_model(window_size: int, n_features: int = 2, horizon: int = 1, hidden_units: int = 128, name: str = "model_multivariate"):
    """
    Model 6: Dense model taking multivariate inputs (Bitcoin Price + Block Reward Halvings).
    """
    _check_tf()
    inputs = layers.Input(shape=(window_size, n_features))
    x = layers.Flatten()(inputs)
    x = layers.Dense(hidden_units, activation="relu")(x)
    outputs = layers.Dense(horizon)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


class NBeatsBlock(_BaseLayer):
    """
    Custom N-BEATS residual block layer (Oreshkin et al., 2019).
    Computes backcast (reconstruction of past input) and forecast (future prediction)
    using shared dense layers followed by theta projection layers.
    """
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        horizon: int,
        n_neurons: int = 128,
        n_layers: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        if TF_AVAILABLE:
            self.hidden = [layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
            self.theta_layer = layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        _check_tf()
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "theta_size": self.theta_size,
            "horizon": self.horizon,
            "n_neurons": self.n_neurons,
            "n_layers": self.n_layers,
        })
        return config


def build_nbeats_model(
    input_size: int,
    horizon: int = 1,
    n_neurons: int = 128,
    n_layers: int = 4,
    n_stacks: int = 3,
    name: str = "model_nbeats"
):
    """
    Model 7: N-BEATS architecture combining stacks of residual blocks.
    Employs doubly residual stacking:
      - Backcast residuals subtract from input signal at each block
      - Forecast signals accumulate additively to form global forecast
    """
    _check_tf()
    theta_size = input_size + horizon
    stack_input = layers.Input(shape=(input_size,), name="stack_input")

    # Initial Block
    first_block = NBeatsBlock(
        input_size=input_size,
        theta_size=theta_size,
        horizon=horizon,
        n_neurons=n_neurons,
        n_layers=n_layers,
        name="NBeatsBlock_0"
    )
    backcast, forecast = first_block(stack_input)
    residuals = layers.subtract([stack_input, backcast], name="subtract_0")

    # Subsequent Doubly Residual Stacks
    for i in range(1, n_stacks):
        block = NBeatsBlock(
            input_size=input_size,
            theta_size=theta_size,
            horizon=horizon,
            n_neurons=n_neurons,
            n_layers=n_layers,
            name=f"NBeatsBlock_{i}"
        )
        backcast, block_forecast = block(residuals)
        residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
        forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

    return tf.keras.Model(inputs=stack_input, outputs=forecast, name=name)


def make_ensemble_preds(ensemble_models: List[Any], data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Model 8: Deep Ensemble with Uncertainty Intervals.
    Computes ensemble mean and 95% confidence bounds (2.5th and 97.5th percentiles).
    """
    preds_list = []
    for model in ensemble_models:
        preds = model.predict(data, verbose=0)
        preds_list.append(preds)
    preds_arr = np.array(preds_list)  # [num_models, num_samples, horizon]
    mean_preds = np.mean(preds_arr, axis=0)
    median_preds = np.median(preds_arr, axis=0)
    lower_bound = np.percentile(preds_arr, 2.5, axis=0)
    upper_bound = np.percentile(preds_arr, 97.5, axis=0)
    return mean_preds, median_preds, lower_bound, upper_bound
