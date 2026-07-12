import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# 1. Page Configuration Setup
st.set_page_config(page_title="Forex Trade Optimizer", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Forex Trade Optimizer")
st.caption("Data-driven predictive timing recommendations tailored to your structural operational cycles.")

# 2. Sidebar Controls
st.sidebar.header("Trade Settings")
home_curr = st.sidebar.selectbox("Your Home Currency (Base)", ["INR", "USD", "EUR", "GBP", "AED"])
trade_curr = st.sidebar.selectbox("Trading Partner Currency", ["USD", "EUR", "JPY", "GBP", "CAD", "INR"])
action = st.sidebar.radio("Your Intended Operation", ["Import (Buy Goods)", "Export (Sell Goods)"])
frequency = st.sidebar.selectbox("Frequency of Commerce", ["Weekly", "Monthly", "Quarterly", "Annually"])

ticker_symbol = f"{home_curr}{trade_curr}=X"

# Map out operational data sizes to allow the pattern engine enough runway
freq_days_map = {
    "Weekly": 90,       
    "Monthly": 180,     
    "Quarterly": 365,    
    "Annually": 730      
}
lookback_days = freq_days_map[frequency]

if home_curr == trade_curr:
    st.error("Home currency and Trade currency cannot be the same. Please select different currencies.")
else:
    # 3. Guarded Data Ingestion Engine
    @st.cache_data(ttl=3600)
    def fetch_forex_data(ticker, days):
        try:
            end_dt = datetime.today()
            start_dt = end_dt - timedelta(days=days)
            df = yf.download(ticker, start=start_dt, end=end_dt, interval="1d", progress=False)
            return df
        except Exception:
            return None

    with st.spinner(f"Ingesting market data..."):
        raw_data = fetch_forex_data(ticker_symbol, lookback_days)
        
    if raw_data is None or raw_data.empty:
        st.warning(f"⚠️ Market connection for {home_curr}/{trade_curr} timed out. Please try switching to another currency pair or refresh the page.")
    else:
        try:
            # Safe column extraction across all variations
            data = pd.DataFrame(index=raw_data.index)
            if 'Close' in raw_data.columns:
                close_series = raw_data['Close']
                if isinstance(close_series, pd.DataFrame):
                    data['Close'] = close_series.iloc[:, 0].values
                else:
                    data['Close'] = close_series.values
            
            if 'Close' not in data.columns or data['Close'].dropna().empty:
                st.warning(f"⚠️ Market values for {home_curr}/{trade_curr} are currently incomplete.")
            else:
                data = data.dropna(subset=['Close']).copy()
                
                # Extract clean string-based context fields
                data['Day_of_Week'] = data.index.day_name()
                data['Day_of_Month'] = data.index.day
                data['Month_Name'] = data.index.strftime('%B')
                
                # Convert explicitly to standard Python types
                prices = [float(x) for x in data['Close'].tolist()]
                current_rate = prices[-1]
                historical_max = max(prices)
                historical_min = min(prices)
                
                # Find exact indices for high/low callouts
                max_idx = data['Close'].idxmax()
                min_idx = data['Close'].idxmin()

                # Generate dynamic indicator lines safely using rolling features
                data['SMA_20'] = data['Close'].rolling(window=min(20, len(data))).mean()
                data['STD_20'] = data['Close'].rolling(window=min(20, len(data))).std()
                data['Upper_Band'] = data['SMA_20'] + (data['STD_20'] * 1.5)
                data['Lower_Band'] = data['SMA_20'] - (data['STD_20'] * 1.5)
                
                # Modern version-agnostic backfill/forwardfill syntax for Pandas 3.0+
                data.bfill(inplace=True)
                data.ffill(inplace=True)

                # 4. Pure Python Statistical Isolation Engine
                if frequency == "Weekly":
                    grouped = data.groupby('Day_of_Week')['Close'].mean()
                    target_day = grouped.idxmax() if action == "Import (Buy Goods)" else grouped.idxmin()
                    target_window = f"Every {target_day}"
                    window_explanation = f"Analysis indicates exchange rates regularly optimize on **{target_day}s** inside your current weekly cycles."
                    matched_dates = data[data['Day_of_Week'] == target_day].index
                    target_idx = matched_dates[-1] if not matched_dates.empty else data.index[-1]
                    
                elif frequency == "Monthly":
                    d1 = data[data['Day_of_Month'] <= 10]['Close'].mean()
                    d2 = data[(data['Day_of_Month'] > 10) & (data['Day_of_Month'] <= 20)]['Close'].mean()
                    d3 = data[data['Day_of_Month'] > 20]['Close'].mean()
                    
                    means = {"Early-Month Cluster (Days 1 to 10)": d1, "Mid-Month Cluster (Days 11 to 20)": d2, "Late-Month Cluster (Days 21 to 31)": d3}
                    means = {k: v for k, v in means.items() if not pd.isna(v)}
                    
                    target_window = max(means, key=means.get) if action == "Import (Buy Goods)" else min(means, key=means.get)
                    window_explanation = f"Historical data isolates a high-efficiency window around the **{target_window}** structural boundary."
                    target_idx = max_idx if action == "Import (Buy Goods)" else min_idx
                    
                elif frequency == "Quarterly":
                    data['Quarter_Week'] = (data.index.dayofyear // 7) % 12 + 1
                    grouped = data.groupby('Quarter_Week')['Close'].mean()
                    target_week = grouped.idxmax() if action == "Import (Buy Goods)" else grouped.idxmin()
                    target_window = f"Week {target_week} of the Quarter"
                    window_explanation = f"Macro trends suggest planning commercial actions during **{target_window}**."
                    target_idx = max_idx if action == "Import (Buy Goods)" else min_idx
                    
                else:  # Annually
                    grouped = data.groupby('Month_Name')['Close'].mean()
                    target_month = grouped.idxmax() if action == "Import (Buy Goods)" else grouped.idxmin()
                    target_window = f"Month of {target_month}"
                    window_explanation = f"Long-term seasonal patterns show optimal settlement margins during the **{target_month}** visual timeline."
                    matched_dates = data[data['Month_Name'] == target_month].index
                    target_idx = matched_dates[-1] if not matched_dates.empty else data.index[-1]

                # 5. UI Layout - Matrix Display
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Current Live Market Rate", value=f"{current_rate:.4f}")
                col2.metric(label="🎯 Recommended Execution Target", value=target_window)
                col3.metric(label="Cycle High / Low Boundaries", value=f"{historical_max:.4f} / {historical_min:.4f}")
                
                st.markdown("---")
                
                # 6. Actionable Recommendation Cards
                st.subheader(f"💡 Strategy Recommendation ({frequency} Commerce Focus)")
                if action == "Import (Buy Goods)":
                    strategy_desc = "Maximize purchasing power by entering the market when home valuation peaks."
                else:
                    strategy_desc = "Maximize settlement margins by converting foreign wire transfers during local currency dips."
                    
                st.write(f"**Core Objective:** {strategy_desc}")
                st.info(f"📅 **Recommended Time Window:** {window_explanation}")
                
                # Process visual alert windows safely using native arrays
                recent_avg = sum(prices[-5:]) / 5 if len(prices) >= 5 else current_rate
                recent_trend_up = current_rate > recent_avg
                
                if action == "Import (Buy Goods)":
                    if recent_trend_up and (current_rate >= historical_max * 0.96):
                        st.success("🟢 **STRATEGY ALERT: FAVORABLE WINDOW OPEN**\n\nThe current rate aligns perfectly with your cycle parameters. Execute your pending trade configurations immediately.")
                    else:
                        st.warning(f"⚠️ **STRATEGY ALERT: DELAY FOR TARGET WINDOW**\n\nConditions are adjusting. Hold non-essential purchases until the market enters your recommended timing slot.")
                else:  # Export
                    if not recent_trend_up and (current_rate <= historical_min * 1.04):
                        st.success("🟢 **STRATEGY ALERT: FAVORABLE WINDOW OPEN**\n\nExchange spreads match your target floor metrics. Process pending foreign invoicing conversions now.")
                    else:
                        st.warning(f"⚠️ **STRATEGY ALERT: DELAY FOR TARGET WINDOW**\n\nHold asset conversion processes if logistics allow until the pair hits your isolated timeframe.")

                # 7. Rich-Packed Dynamic Visualization Segment
                st.subheader(f"📊 Market Structure Analysis ({frequency} Context Timeline)")
                
                fig = go.Figure()
                
                # Line 1: Main Exchange Closing Price
                fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Actual Exchange Close', line=dict(color='#1f77b4', width=2.5)))
                
                # Line 2: 20-Day Simple Moving Average Trend Line
                fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='20-Day Trend Mean', line=dict(color='#ff7f0e', width=1.5, dash='dash')))
                
                # Line 3 & 4: Structural Corridor Bands
                fig.add_trace(go.Scatter(x=data.index, y=data['Upper_Band'], name='Upper Valuation Band', line=dict(color='rgba(44, 160, 44, 0.4)', width=1)))
                fig.add_trace(go.Scatter(x=data.index, y=data['Lower_Band'], name='Lower Valuation Band', line=dict(color='rgba(214, 39, 40, 0.4)', width=1), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.05)'))
                
                # Point Callout 1: Historical Cycle Peak Marker
                fig.add_trace(go.Scatter(x=[max_idx], y=[historical_max], mode='markers', name='Cycle Absolute High', marker=dict(color='#2ca02c', size=10, symbol='triangle-up')))
                
                # Point Callout 2: Historical Cycle Floor Marker
                fig.add_trace(go.Scatter(x=[min_idx], y=[historical_min], mode='markers', name='Cycle Absolute Low', marker=dict(color='#d62728', size=10, symbol='triangle-down')))
                
                # Point Callout 3: Golden Target Target Zone Star Marker
                fig.add_trace(go.Scatter(x=[target_idx], y=[data['Close'].loc[target_idx]], mode='markers', name='Isolated Optimal Cycle Node', marker=dict(color='#e377c2', size=14, symbol='star')))

                # Custom Callout Annotation Card
                fig.add_annotation(
                    x=target_idx, 
                    y=float(data['Close'].loc[target_idx]), 
                    text=f"Optimal Target Timeframe ({target_window})", 
                    showarrow=True, 
                    arrowhead=2, 
                    arrowcolor="#e377c2", 
                    font=dict(color="#ffffff", size=11),
                    bgcolor="#e377c2",
                    bordercolor="#e377c2",
                    borderwidth=1,
                    ax=0, 
                    ay=-40
                )
                
                fig.update_layout(
                    xaxis_title="Timeline Context",
                    yaxis_title=f"1 {home_curr} to {trade_curr} Value",
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=520,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"An unexpected data rendering condition occurred: {e}. Re-syncing currency metrics.")