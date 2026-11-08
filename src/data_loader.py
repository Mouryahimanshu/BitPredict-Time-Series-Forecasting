"""
Data loading and preprocessing module for Bitcoin Time Series Forecasting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import minmax_scale

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv")


def load_bitcoin_data(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Loads Bitcoin historical price data, parses dates, and formats price.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
        
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col=["Date"])
    prices = pd.DataFrame(df["Closing Price (USD)"]).rename(columns={"Closing Price (USD)": "Price"})
    prices = prices.asfreq("D")
    prices["Price"] = prices["Price"].ffill()
    return prices


def add_block_reward_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends historical Bitcoin block reward halving amounts to the dataframe.
    Halving events:
      - Genesis (2009-01-03): 50 BTC
      - 1st Halving (2012-11-28): 25 BTC
      - 2nd Halving (2016-07-09): 12.5 BTC
      - 3rd Halving (2020-05-11): 6.25 BTC
      - 4th Halving (2024-04-19): 3.125 BTC
    """
    df = prices_df.copy()
    
    halvings = [
        (np.datetime64("2009-01-03"), 50.0),
        (np.datetime64("2012-11-28"), 25.0),
        (np.datetime64("2016-07-09"), 12.5),
        (np.datetime64("2020-05-11"), 6.25),
        (np.datetime64("2024-04-19"), 3.125),
    ]
    
    df["block_reward"] = 25.0  # initial default for 2013 dataset start
    
    for date, reward in halvings:
        df.loc[df.index >= pd.Timestamp(date), "block_reward"] = reward
        
    return df


def get_scaled_multivariate_data(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates normalized multivariate features combining Price and block_reward.
    """
    df = add_block_reward_features(prices_df)
    scaled = minmax_scale(df[["Price", "block_reward"]])
    return pd.DataFrame(scaled, columns=["Price_scaled", "block_reward_scaled"], index=df.index)


def create_turkey_data(prices_df: pd.DataFrame, drop_factor: float = 0.9, drop_days: int = 1) -> pd.DataFrame:
    """
    Creates synthetic 'Turkey Problem' (Black Swan) dataset by introducing a catastrophic
    drop in the final test period to assess model fragility under unprecedented market shifts.
    """
    turkey_df = prices_df.copy()
    test_split_idx = int(len(turkey_df) * 0.8)
    
    # Simulate a sudden market collapse midway through test set
    crash_idx = test_split_idx + (len(turkey_df) - test_split_idx) // 2
    turkey_df.iloc[crash_idx:crash_idx + drop_days, 0] = turkey_df.iloc[crash_idx - 1, 0] * (1 - drop_factor)
    return turkey_df
