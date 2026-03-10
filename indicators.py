"""
indicators.py
計算技術指標：均線、MACD、RSI
"""

import pandas as pd
import numpy as np
from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL, RSI_PERIOD


def calc_ma(df, periods=[5, 10, 20, 60]):
    """計算多條均線"""
    for p in periods:
        df[f"ma{p}"] = df["close"].rolling(window=p).mean().round(2)
    return df


def calc_ema(series, period):
    """計算指數移動平均"""
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(df, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    """計算 MACD（依設定：6, 13, 9）"""
    ema_fast = calc_ema(df["close"], fast)
    ema_slow = calc_ema(df["close"], slow)
    df["macd"] = (ema_fast - ema_slow).round(4)
    df["macd_signal"] = calc_ema(df["macd"], signal).round(4)
    df["macd_hist"] = (df["macd"] - df["macd_signal"]).round(4)
    return df


def calc_rsi(df, period=RSI_PERIOD):
    """計算 RSI"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).round(2)
    return df


def calc_volume_ma(df, period=5):
    """計算量均線（判斷量是否異常）"""
    df["vol_ma5"] = df["volume"].rolling(window=period).mean().round(0)
    df["vol_ratio"] = (df["volume"] / df["vol_ma5"]).round(2)  # 量比
    return df


def add_all_indicators(df):
    """一次加入所有技術指標"""
    if df.empty or len(df) < 20:
        return df
    df = calc_ma(df)
    df = calc_macd(df)
    df = calc_rsi(df)
    df = calc_volume_ma(df)
    return df


def get_latest_signals(df):
    """
    取得最新一日的技術訊號摘要
    回傳一個 dict，供 AI 分析使用
    """
    if df.empty:
        return {}

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last

    signals = {}

    # 基本價格
    signals["date"] = str(last.get("date", ""))[:10]
    signals["close"] = float(last.get("close", 0))
    signals["open"] = float(last.get("open", 0))
    signals["high"] = float(last.get("high", 0))
    signals["low"] = float(last.get("low", 0))
    signals["volume(張)"] = int(last.get("volume", 0)) // 1000

    # 漲跌
    change = last["close"] - prev["close"]
    change_pct = (change / prev["close"] * 100) if prev["close"] else 0
    signals["change"] = round(float(change), 2)
    signals["change_pct"] = round(float(change_pct), 2)

    # 量比（今日量 vs 5日均量）
    signals["vol_ratio"] = float(last.get("vol_ratio", 1))

    # 均線位置
    for ma in [5, 10, 20, 60]:
        val = last.get(f"ma{ma}")
        signals[f"ma{ma}"] = round(float(val), 2) if pd.notna(val) else None

    # 均線多空排列
    ma5 = signals.get("ma5")
    ma20 = signals.get("ma20")
    ma60 = signals.get("ma60")
    if ma5 and ma20 and ma60:
        if ma5 > ma20 > ma60:
            signals["ma_alignment"] = "多頭排列"
        elif ma5 < ma20 < ma60:
            signals["ma_alignment"] = "空頭排列"
        else:
            signals["ma_alignment"] = "糾結整理"
    else:
        signals["ma_alignment"] = "資料不足"

    # 收盤與均線關係
    close = signals["close"]
    signals["above_ma5"] = close > ma5 if ma5 else None
    signals["above_ma20"] = close > ma20 if ma20 else None
    signals["above_ma60"] = close > ma60 if ma60 else None

    # MACD
    signals["macd"] = float(last.get("macd", 0)) if pd.notna(last.get("macd")) else None
    signals["macd_signal"] = float(last.get("macd_signal", 0)) if pd.notna(last.get("macd_signal")) else None
    signals["macd_hist"] = float(last.get("macd_hist", 0)) if pd.notna(last.get("macd_hist")) else None
    prev_hist = float(prev.get("macd_hist", 0)) if pd.notna(prev.get("macd_hist")) else 0
    curr_hist = signals["macd_hist"] or 0
    if curr_hist > 0 and curr_hist > prev_hist:
        signals["macd_status"] = "多頭擴張"
    elif curr_hist > 0 and curr_hist < prev_hist:
        signals["macd_status"] = "多頭收斂"
    elif curr_hist < 0 and curr_hist < prev_hist:
        signals["macd_status"] = "空頭擴張"
    elif curr_hist < 0 and curr_hist > prev_hist:
        signals["macd_status"] = "空頭收斂"
    else:
        signals["macd_status"] = "持平"

    # RSI
    signals["rsi"] = float(last.get("rsi", 50)) if pd.notna(last.get("rsi")) else None
    rsi = signals["rsi"] or 50
    if rsi >= 70:
        signals["rsi_status"] = "超買區"
    elif rsi <= 30:
        signals["rsi_status"] = "超賣區"
    elif rsi >= 60:
        signals["rsi_status"] = "偏強"
    elif rsi <= 40:
        signals["rsi_status"] = "偏弱"
    else:
        signals["rsi_status"] = "中性"

    return signals
