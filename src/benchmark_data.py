"""
Benchmark metrics and experiment logs directly extracted from the project notebook.
Contains full comparison table, model parameters, and experimental findings.
"""

import pandas as pd

BENCHMARK_RESULTS = {
    "Model 0: Naive Baseline (Shift 1)": {
        "mae": 567.9802,
        "mse": 1147547.0,
        "rmse": 1071.2362,
        "mape": 2.5165,
        "mase": 1.0014,
        "window": 1,
        "horizon": 1,
        "description": "Baseline: y_t = y_{t-1}. Predicts previous day's price."
    },
    "Model 1: Dense (w=7, h=1)": {
        "mae": 562.5154,
        "mse": 1149832.5,
        "rmse": 1072.3024,
        "mape": 2.5092,
        "mase": 0.9899,
        "window": 7,
        "horizon": 1,
        "description": "Dense MLP with 128 hidden units using past 7 days."
    },
    "Model 2: Dense (w=30, h=1)": {
        "mae": 619.8304,
        "mse": 1302923.4,
        "rmse": 1141.4567,
        "mape": 2.7990,
        "mase": 1.0854,
        "window": 30,
        "horizon": 1,
        "description": "Dense MLP with 128 hidden units using past 30 days."
    },
    "Model 3: Dense (w=30, h=7)": {
        "mae": 1265.5134,
        "mse": 5568592.5,
        "rmse": 2359.7866,
        "mape": 5.6761,
        "mase": 2.2560,
        "window": 30,
        "horizon": 7,
        "description": "Multi-horizon dense model forecasting 7 days ahead."
    },
    "Model 4: Conv1D (w=7, h=1)": {
        "mae": 571.2466,
        "mse": 1180574.1,
        "rmse": 1086.5422,
        "mape": 2.5616,
        "mase": 1.0053,
        "window": 7,
        "horizon": 1,
        "description": "1D Temporal Convolutional network with causal padding."
    },
    "Model 5: RNN / LSTM (w=7, h=1)": {
        "mae": 583.6704,
        "mse": 1210980.6,
        "rmse": 1100.4457,
        "mape": 2.6235,
        "mase": 1.0272,
        "window": 7,
        "horizon": 1,
        "description": "Single-layer LSTM with 128 hidden units."
    },
    "Model 6: Multivariate Dense (w=7, h=1)": {
        "mae": 565.6340,
        "mse": 1155174.4,
        "rmse": 1074.7904,
        "mape": 2.5343,
        "mase": 0.9955,
        "window": 7,
        "horizon": 1,
        "description": "Dense model taking Bitcoin price + block reward halving schedule."
    },
    "Model 7: N-BEATS (w=7, h=1)": {
        "mae": 576.8463,
        "mse": 1185465.9,
        "rmse": 1088.7910,
        "mape": 2.6435,
        "mase": 1.0152,
        "window": 7,
        "horizon": 1,
        "description": "Neural Basis Expansion with backcast and forecast residual blocks."
    },
    "Model 8: Ensemble (Stacking 5 Models)": {
        "mae": 646.8053,
        "mse": 4876337.5,
        "rmse": 2208.2429,
        "mape": 21.5847,
        "mase": 26.5544,
        "window": 7,
        "horizon": 1,
        "description": "Deep ensemble combining MAE, MSE, and MAPE trained models with prediction intervals."
    },
    "Model 9: Turkey Model (Black Swan Shock)": {
        "mae": 642.7346,
        "mse": 4902171.0,
        "rmse": 2214.0847,
        "mape": 21.6207,
        "mase": 26.5509,
        "window": 7,
        "horizon": 1,
        "description": "Tested on Taleb's Turkey problem dataset (sudden 90% plunge simulating regime shift)."
    }
}


def get_benchmark_dataframe() -> pd.DataFrame:
    """
    Returns the benchmark comparison table as a formatted pandas DataFrame.
    """
    df = pd.DataFrame(BENCHMARK_RESULTS).T
    cols = ["mae", "mse", "rmse", "mape", "mase", "window", "horizon", "description"]
    return df[cols]
