"""
Time Series Forecasting Project: BitPredict
A comprehensive deep learning time series forecasting suite for Bitcoin prices built with TensorFlow / Keras.
"""

from src.models_tf import (
    NaiveModel,
    build_dense_model,
    build_conv1d_model,
    build_lstm_model,
    build_multivariate_dense_model,
    NBeatsBlock,
    build_nbeats_model,
    make_ensemble_preds,
)

__version__ = "1.0.0"
