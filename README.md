# DS-Project Task
#Trader Behavior Insights - objective is to explore the relationship between trader performance and market 
#sentiment, uncover hidden patterns, and deliver insights that can drive smarter trading 
#strategies.


import pandas as pd
import numpy as np

# Load data
trades_df = pd.read_csv(r"C:\Users\SHUBHAM\Downloads\hyperliquid_trades.csv")
fg_df = pd.read_csv(r"C:\Users\SHUBHAM\Downloads\fear_greed.csv")

print("Trades columns:", trades_df.columns.tolist())
print("FearGreed columns:", fg_df.columns.tolist())
print("\nTrades sample:\n", trades_df.head(2))
print("\nFearGreed sample:\n", fg_df.head(2))

# Auto-detect time/date columns for trades
time_cols = ['time', 'timestamp', 'Time', 'Timestamp']
date_col_fg = ['Date', 'date', 'timestamp', 'Timestamp']

time_col = next((col for col in time_cols if col in trades_df.columns), None)
if time_col is None:
    print("No time column found! Available:", trades_df.select_dtypes(include=['datetime64', 'int64', 'float64']).columns.tolist())
    raise KeyError("Add time column manually or check CSV")

print(f"Using time column: '{time_col}'")

# Preprocess trades time
if trades_df[time_col].dtype in ['int64', 'float64']:
    trades_df['time'] = pd.to_datetime(trades_df[time_col], unit='ms', errors='coerce')
else:
    trades_df['time'] = pd.to_datetime(trades_df[time_col], errors='coerce')
trades_df['date'] = trades_df['time'].dt.date

# Auto-detect FG date column
fg_date_col = next((col for col in date_col_fg if col in fg_df.columns), None)
if fg_date_col is None:
    raise KeyError("No date column in FearGreed CSV")
print(f"Using FG date column: '{fg_date_col}'")

fg_df['date'] = pd.to_datetime(fg_df[fg_date_col], errors='coerce').dt.date

# Ensure Classification exists (common names: Classification, value_classification, fear_greed_category)
class_cols = ['Classification', 'classification', 'fear_greed_category', 'FEAR_GREED_CATEGORY']
class_col = next((col for col in class_cols if col in fg_df.columns), None)
if class_col:
    fg_df['Classification'] = fg_df[class_col].astype(str)
else:
    print("No classification column; using numeric if available.")
    fg_df['Classification'] = pd.cut(fg_df[fg_date_col].astype(float), bins=[0,25,50,75,100], 
                                     labels=['Extreme Fear', 'Fear', 'Greed', 'Extreme Greed'])

# Drop invalids
trades_df = trades_df.dropna(subset=['date'])
fg_df = fg_df.dropna(subset=['date'])

print("Trades date range:", trades_df['date'].min(), "to", trades_df['date'].max())
print("FG date range:", fg_df['date'].min(), "to", fg_df['date'].max())

# Merge (ensure 'closedPnL' exists; alt: 'pnl', 'PnL')
pnl_col = next((col for col in trades_df.columns if 'pnl' in col.lower() or 'pnl' in col.lower()), 'closedPnL')
print(f"Using PnL column: '{pnl_col}'")

merged = pd.merge(trades_df, fg_df, on='date', how='inner')
print(f"Merged shape: {merged.shape}")

if merged.empty:
    print("No overlaps. Try weekly merge:")
    trades_df['week'] = trades_df['time'].dt.isocalendar().week
    fg_df['week'] = fg_df['date'].dt.isocalendar().week
    merged = pd.merge(trades_df, fg_df, on='week', how='inner')
    print(f"Weekly merged shape: {merged.shape}")
else:
    # Analysis
    pnl_by_sentiment = merged.groupby('Classification')[pnl_col].agg(['mean', 'count', 'std']).round(4)
    print("\nPnL by Sentiment:\n", pnl_by_sentiment)
    
    # Correlation
    sentiment_num = merged['Classification'].map({'Fear':0, 'Extreme Fear':0, 'Greed':1, 'Extreme Greed':1}).fillna(0.5)
    corr = sentiment_num.corr(merged[pnl_col].fillna(0))
    print(f"\nPnL-Sentiment Correlation: {corr:.4f}")
