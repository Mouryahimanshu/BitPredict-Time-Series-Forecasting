"""
BitPredict: Interactive Streamlit Dashboard for Bitcoin Time Series Forecasting.
Showcases 10 deep learning forecasting models, evaluation metrics, interactive charts,
model architectures, prediction intervals, and the Turkey Problem analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import load_bitcoin_data, add_block_reward_features, create_turkey_data
from src.benchmark_data import get_benchmark_dataframe, BENCHMARK_RESULTS
from src.metrics import evaluate_preds
from src.models_torch import NaiveModel

# Page configuration
st.set_page_config(
    page_title="BitPredict | Deep Learning Time Series Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F7931A 0%, #FFB74D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #2d3139;
        text-align: center;
    }
    .kpi-title { font-size: 0.85rem; color: #a0aec0; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #f7931a; margin-top: 4px; }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        background: #2b313e;
        color: #f7931a;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def get_data():
    df = load_bitcoin_data()
    df_multivariate = add_block_reward_features(df)
    benchmark_df = get_benchmark_dataframe()
    return df, df_multivariate, benchmark_df

df_prices, df_multi, benchmark_df = get_data()

# Sidebar
st.sidebar.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png?v=032", width=70)
st.sidebar.title("BitPredict Navigator")
st.sidebar.caption("Milestone Project 3: Time Series Forecasting")

nav_choice = st.sidebar.radio(
    "Explore Sections:",
    [
        "📊 Overview & Bitcoin History",
        "🏆 Model Benchmark Leaderboard",
        "🔬 Deep Learning Architectures",
        "🖼️ Visual Gallery & Forecasts",
        "🦃 The Turkey Problem (Black Swan)",
        "🔮 Interactive Forecaster"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Key Technologies:**
- PyTorch 2.14 / TensorFlow
- Pandas & NumPy
- Plotly & Streamlit
- N-BEATS Deep Architecture
""")

# ==========================================
# SECTION 1: OVERVIEW & BITCOIN HISTORY
# ==========================================
if nav_choice == "📊 Overview & Bitcoin History":
    st.markdown('<div class="main-header">BitPredict: Time Series Forecasting Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluating Deep Learning Architectures for Financial Time Series & Crypto Market Forecasting</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">Time Horizon Analyzed</div>
            <div class="kpi-value">{len(df_prices):,} Days</div>
            <span class="badge">Oct 2013 - May 2021</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="kpi-title">All-Time High (In Dataset)</div>
            <div class="kpi-value">${df_prices['Price'].max():,.2f}</div>
            <span class="badge">April 2021 Peak</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-title">Best Model MAE</div>
            <div class="kpi-value">$562.51</div>
            <span class="badge">Dense (w=7, h=1)</span>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="kpi-title">Baseline Benchmark</div>
            <div class="kpi-value">$567.98</div>
            <span class="badge">Naive Forecast ($y_t = y_{t-1}$)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Bitcoin Historical Price & Halving Cycles")
    
    # Plotly interactive historical chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_prices.index,
        y=df_prices['Price'],
        mode='lines',
        name='BTC Closing Price',
        line=dict(color='#F7931A', width=2)
    ))

    # Add Halving annotations
    halvings = [
        ("2016-07-09", "2nd Halving (12.5 BTC)", 650.0),
        ("2020-05-11", "3rd Halving (6.25 BTC)", 8600.0)
    ]
    for date_str, label, price_y in halvings:
        fig.add_vline(x=date_str, line_width=1.5, line_dash="dash", line_color="#00E676")
        fig.add_annotation(
            x=date_str,
            y=price_y,
            text=label,
            showarrow=True,
            arrowhead=2,
            arrowcolor="#00E676",
            arrowsize=1,
            arrowwidth=1.5,
            bgcolor="#1A1C24",
            bordercolor="#00E676",
            font=dict(color="#00E676", size=10)
        )

    fig.update_layout(
        title="Bitcoin Daily Closing Price (USD) with Halving Dates",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=500,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(r"""
    #### 💡 Project Abstract & Research Questions
    1. **Can Deep Learning Outperform a Naive Baseline?**  
       In financial time series, prices frequently follow an approximate *random walk* ($y_t = y_{t-1} + \epsilon$). Can neural networks (Dense, 1D CNNs, LSTMs, N-BEATS) reliably beat the naive baseline?
    2. **Lookback Windows vs Forecast Horizons:**  
       Does providing more historical context (30 days vs 7 days) improve predictive accuracy, or does it inject redundant noise?
    3. **Exogenous Features (Halving Cycles):**  
       Does incorporating block reward issuance schedules reduce generalization error?
    4. **Uncertainty & Ensembling:**  
       How do ensemble prediction intervals handle volatility clusters?
    5. **The Turkey Problem (Black Swan Fragility):**  
       What happens to deep learning models when exposed to unprecedented market structural shifts?
    """)

# ==========================================
# SECTION 2: MODEL BENCHMARK LEADERBOARD
# ==========================================
elif nav_choice == "🏆 Model Benchmark Leaderboard":
    st.markdown('<div class="main-header">Model Benchmark & Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Systematic performance comparison across 10 forecasting architectures</div>', unsafe_allow_html=True)

    # Leaderboard Table
    sorted_df = benchmark_df.sort_values(by="mae").copy()
    
    st.dataframe(
        sorted_df.style.format({
            "mae": "${:,.2f}",
            "mse": "{:,.1f}",
            "rmse": "${:,.2f}",
            "mape": "{:.2f}%",
            "mase": "{:.4f}",
            "window": "{:d}",
            "horizon": "{:d}"
        }).highlight_min(subset=["mae", "rmse", "mape", "mase"], color="#1b4d3e"),
        use_container_width=True
    )

    col1, col2 = st.columns(2)
    with col1:
        # Bar Chart of MAE
        fig_mae = px.bar(
            sorted_df.reset_index(),
            x="index",
            y="mae",
            color="mae",
            color_continuous_scale="Viridis_r",
            labels={"index": "Model", "mae": "Mean Absolute Error ($)"},
            title="Model Ranking by MAE (Lower is Better)"
        )
        fig_mae.update_layout(template="plotly_dark", xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_mae, use_container_width=True)

    with col2:
        # Bar Chart of MASE
        fig_mase = px.bar(
            sorted_df[sorted_df["mase"] < 5].reset_index(),  # Filter out extreme shock models for clarity
            x="index",
            y="mase",
            color="mase",
            color_continuous_scale="Tealgrn_r",
            labels={"index": "Model", "mase": "MASE (vs Naive Baseline)"},
            title="MASE Relative to Naive Benchmark (MASE < 1 beats Naive)"
        )
        fig_mase.add_hline(y=1.0, line_dash="dash", line_color="#FF5252", annotation_text="Naive Threshold (1.0)")
        fig_mase.update_layout(template="plotly_dark", xaxis_tickangle=-45, height=450)
        st.plotly_chart(fig_mase, use_container_width=True)

    st.info("""
    **Key Theoretical Takeaway (The Efficient Market Dilemma):**  
    Notice that the **Naive Forecast** ($MAE = $567.98) places 3rd among all models, outperforming LSTMs ($583.67), Conv1D ($571.25), and N-BEATS ($576.85).  
    Only a simple Dense model with a short window (w=7) slightly edged past it ($562.52, MASE=0.9899).  
    This confirms the classic **Random Walk Hypothesis** in quantitative finance: past prices largely incorporate current information, making 1-step ahead price levels strongly dependent on the most recent tick.
    """)

# ==========================================
# SECTION 3: DEEP LEARNING ARCHITECTURES
# ==========================================
elif nav_choice == "🔬 Deep Learning Architectures":
    st.markdown('<div class="main-header">Deep Learning Architectures Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detailed structural breakdown of the implemented models</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧠 1. Dense & Lookback Windows",
        "🌊 2. Temporal Conv1D & LSTM",
        "⚡ 3. N-BEATS Architecture",
        "🎯 4. Multi-Horizon Forecasting",
        "🛡️ 5. Deep Ensembles & Uncertainty"
    ])

    with tab1:
        st.markdown("### Dense Feed-Forward Neural Networks")
        st.write("""
        We evaluated standard Multi-Layer Perceptrons (MLPs) across two lookback windows:
        - **Model 1 (Window=7, Horizon=1):** Inputs the previous 7 days to predict day 8. $MAE = \\$562.51$
        - **Model 2 (Window=30, Horizon=1):** Inputs the previous 30 days to predict day 31. $MAE = \\$619.83$
        """)
        st.warning("""
        **Observation:** Increasing the lookback window from 7 to 30 days **worsened** test MAE by +$57.32 (10.2% error increase).
        In volatile crypto markets, distant historical prices act as non-stationary noise rather than predictive signal.
        """)

    with tab2:
        st.markdown("### Temporal Convolutional Networks (Conv1D) vs LSTM")
        st.write("""
        - **Conv1D (Model 4):** Uses 1D causal convolutions (kernel_size=5, filters=128, padding='causal') to extract local temporal patterns without future leakage.
        - **LSTM (Model 5):** Recurrent neural network with memory cells capturing sequential state transitions.
        """)
        st.markdown("""
        | Model | Architecture Highlights | Parameters | Test MAE |
        |---|---|---|---|
        | **Conv1D** | Causal Conv1D(128, 5) -> GlobalPool -> Dense(1) | ~17K | **$571.25** |
        | **LSTM** | LSTM(128) -> Dense(1) | ~67K | **$583.67** |
        """)

    with tab3:
        st.markdown("### N-BEATS (Neural Basis Expansion Analysis for Time Series)")
        st.write("""
        N-BEATS (Oreshkin et al., 2019) is a state-of-the-art pure deep learning architecture designed for time series forecasting without recurrent or attentional components.
        """)
        if os.path.exists("assets/14_nbeats_architecture.png"):
            st.image("assets/14_nbeats_architecture.png", caption="N-BEATS Architecture: Stacks of Residual Blocks with Backcast and Forecast expansion", use_container_width=True)
        st.markdown("""
        **Core N-BEATS Principles:**
        1. **Doubly Residual Stacking:**
           - Each block outputs a *backcast* $\\hat{x}$ (reconstruction of past input) and a *forecast* $\\hat{y}$ (prediction of future).
           - The backcast is subtracted from the input signal: $x_{l} = x_{l-1} - \\hat{x}_{l-1}$.
           - The forecast signals are summed additively across all blocks: $\\hat{y} = \\sum_{l} \\hat{y}_{l}$.
        2. **Interpretable Basis:** Allows decomposition into trend and seasonality polynomials.
        """)

    with tab4:
        st.markdown("### Multi-Horizon Forecasting (Horizon = 7 Days)")
        st.write("""
        Model 3 evaluates predicting a full 7-day vector ahead ($y_{t+1}, \\dots, y_{t+7}$) simultaneously from 30 historical days.
        """)
        if os.path.exists("assets/12_model_3_mae_per_horizon_day.png"):
            st.image("assets/12_model_3_mae_per_horizon_day.png", caption="MAE Growth per Forecast Horizon Day", use_container_width=True)
        st.markdown("""
        **Key Finding:** Error grows approximately monotonically with horizon distance:
        - Day 1 MAE: ~\\$580
        - Day 4 MAE: ~\\$1,150
        - Day 7 MAE: ~\\$1,850
        """)

    with tab5:
        st.markdown("### Deep Ensembles & Predictive Uncertainty Intervals")
        st.write("""
        Point forecasts often give a false sense of precision in financial applications.  
        Model 8 creates an ensemble of diverse models trained with different random seeds and objective functions (MAE, MSE, MAPE) to calculate 95% confidence bounds (2.5th and 97.5th percentiles).
        """)
        if os.path.exists("assets/15_ensemble_uncertainty_prediction_intervals.png"):
            st.image("assets/15_ensemble_uncertainty_prediction_intervals.png", caption="Ensemble Predictions with 95% Uncertainty Prediction Intervals", use_container_width=True)

# ==========================================
# SECTION 4: VISUAL GALLERY
# ==========================================
elif nav_choice == "🖼️ Visual Gallery & Forecasts":
    st.markdown('<div class="main-header">Project Visual Gallery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive figures and prediction curves generated during experimental runs</div>', unsafe_allow_html=True)

    assets = sorted([f for f in os.listdir("assets") if f.endswith(".png")]) if os.path.exists("assets") else []

    if not assets:
        st.warning("No generated assets found in assets/ directory.")
    else:
        for i in range(0, len(assets), 2):
            cols = st.columns(2)
            with cols[0]:
                img_path = os.path.join("assets", assets[i])
                st.image(img_path, caption=assets[i].replace(".png", "").replace("_", " ").title(), use_container_width=True)
            if i + 1 < len(assets):
                with cols[1]:
                    img_path2 = os.path.join("assets", assets[i + 1])
                    st.image(img_path2, caption=assets[i + 1].replace(".png", "").replace("_", " ").title(), use_container_width=True)

# ==========================================
# SECTION 5: THE TURKEY PROBLEM (BLACK SWAN)
# ==========================================
elif nav_choice == "🦃 The Turkey Problem (Black Swan)":
    st.markdown('<div class="main-header">The Turkey Problem & Black Swan Fragility</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Why time series models fail when historical distributions undergo structural breaks</div>', unsafe_allow_html=True)

    st.markdown("""
    #### 📖 Concept Background: Nassim Nicholas Taleb's Turkey Problem
    > *"Consider a turkey that is fed every day. Every single feeding will firm up the bird's belief that it is the general rule of life to be fed every day by friendly members of the human race... On the afternoon of the Wednesday before Thanksgiving, something unexpected will happen to the turkey. It will incur a revision of belief."*  
    > — **Nassim Nicholas Taleb, *The Black Swan***

    In time series forecasting, models trained purely on continuous historical upward trends suffer from the exact same epistemic trap.
    """)

    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("assets/17_turkey_problem_crash_data.png"):
            st.image("assets/17_turkey_problem_crash_data.png", caption="Synthetic Black Swan Shock Injected into Test Set", use_container_width=True)
    with col2:
        if os.path.exists("assets/18_turkey_model_crash_predictions.png"):
            st.image("assets/18_turkey_model_crash_predictions.png", caption="Model Lagging Behind the Severe Crash (Autoregressive Trap)", use_container_width=True)

    st.error("""
    **Empirical Findings:**
    - When exposed to an unprecedented crash, model MAPE explodes from **2.5%** to **21.6%**, and MASE deteriorates from **0.99** to **26.55**.
    - The model continues predicting high prices during the initial drop because its autoregressive lookback window still contains high price values.
    - **Takeaway for Practitioners:** Deep learning models cannot anticipate structural breaks, black swans, regulatory bans, or protocol exploits that have zero representation in training history.
    """)

# ==========================================
# SECTION 6: INTERACTIVE FORECASTER
# ==========================================
elif nav_choice == "🔮 Interactive Forecaster":
    st.markdown('<div class="main-header">Interactive Sandbox Forecaster</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Test sliding window forecasts on recent historical Bitcoin data</div>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        window_size = st.slider("Lookback Window (Days):", min_value=3, max_value=60, value=7)
    with col_ctrl2:
        forecast_horizon = st.slider("Forecast Horizon (Days):", min_value=1, max_value=14, value=1)

    prices = df_prices["Price"].to_numpy()
    last_window = prices[-window_size:]

    naive = NaiveModel(horizon=forecast_horizon)
    naive_pred = naive.predict(last_window)

    st.markdown("### 🔍 Most Recent Historical Window")
    recent_dates = df_prices.index[-window_size:]
    df_recent = pd.DataFrame({"Date": recent_dates, "Price (USD)": last_window})
    st.dataframe(df_recent.style.format({"Price (USD)": "${:,.2f}"}), use_container_width=True)

    st.markdown("### 🎯 Baseline Forecast for Next Steps")
    future_dates = [df_prices.index[-1] + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    df_future = pd.DataFrame({
        "Forecast Date": future_dates,
        "Naive Forecast ($)": naive_pred,
        "Simulated 95% Lower Bound ($)": naive_pred * 0.95,
        "Simulated 95% Upper Bound ($)": naive_pred * 1.05
    })
    st.dataframe(df_future.style.format({
        "Naive Forecast ($)": "${:,.2f}",
        "Simulated 95% Lower Bound ($)": "${:,.2f}",
        "Simulated 95% Upper Bound ($)": "${:,.2f}"
    }), use_container_width=True)

    # Plot
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(
        x=df_recent["Date"],
        y=df_recent["Price (USD)"],
        mode="lines+markers",
        name="Historical Window",
        line=dict(color="#F7931A", width=3)
    ))
    fig_sim.add_trace(go.Scatter(
        x=df_future["Forecast Date"],
        y=df_future["Naive Forecast ($)"],
        mode="lines+markers",
        name="Predicted Horizon",
        line=dict(color="#00E676", width=3, dash="dot")
    ))
    fig_sim.update_layout(template="plotly_dark", height=400, title="Interactive Forecast Projection")
    st.plotly_chart(fig_sim, use_container_width=True)
