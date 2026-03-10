"""
strategies/classic.py
策略 A-F：經典選股策略（從 screener.py 搬出）。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

from strategies import register, StrategyInfo
from strategies._helpers import (
    HIGH_VOL_INDUSTRIES,
    AI_INDUSTRIES,
    check_consecutive_buying,
    check_latest_day_buying,
    detect_swing_highs_lows,
)


# ═══════════════════════════════════════════════════════════
# 策略 A：均線糾結起漲 (Spring Compression)
# ═══════════════════════════════════════════════════════════

def strategy_a_spring(df: pd.DataFrame, industry: str = "", **kwargs) -> dict | None:
    """
    均線糾結起漲：MA5 / MA10 / MA20 三線收斂後帶量突破
    條件：
    1. MA5、MA10、MA20 三者離散度 < 閾值（一般 1.5%，高波動產業 3%）
    2. 收盤 > MA5 且 成交量 > VMA5 * 1.5（起漲訊號）
    3. MA20 向上趨勢（MA20[-1] > MA20[-5]）
    """
    if len(df) < 30:
        return None

    last = df.iloc[-1]
    ma5 = last.get("s_ma5")
    ma10 = last.get("s_ma10")
    ma20 = last.get("s_ma20")
    close = last["close"]
    volume = last["volume"]
    vol_ma5 = last.get("s_vol_ma5")

    if any(pd.isna(v) for v in [ma5, ma10, ma20, vol_ma5]) or ma20 == 0:
        return None

    # 1. 三線糾結判定：高波動產業放寬至 3%，其餘 1.5%
    threshold = 0.03 if industry in HIGH_VOL_INDUSTRIES else 0.015
    ma_max = max(ma5, ma10, ma20)
    ma_min = min(ma5, ma10, ma20)
    compression = (ma_max - ma_min) / ma20
    if compression >= threshold:
        return None

    # 2. 起漲判定
    if close <= ma5 or volume <= vol_ma5 * 1.5:
        return None

    # 3. MA20 向上趨勢
    if len(df) < 6:
        return None
    ma20_now = df["s_ma20"].iloc[-1]
    ma20_5ago = df["s_ma20"].iloc[-5]
    if pd.isna(ma20_5ago) or ma20_now <= ma20_5ago:
        return None

    return {
        "strategy": "A",
        "label": "均線糾結起漲",
        "compression_pct": round(compression * 100, 2),
        "ma5": round(float(ma5), 2),
        "ma10": round(float(ma10), 2),
        "ma20": round(float(ma20), 2),
        "ma20_slope": round(float(ma20_now - ma20_5ago), 2),
        "vol_ratio": round(float(volume / vol_ma5), 2),
    }


# ═══════════════════════════════════════════════════════════
# 策略 B：多頭續強型 (Trend Following)
# ═══════════════════════════════════════════════════════════

def strategy_b_trend(df: pd.DataFrame, **kwargs) -> dict | None:
    """
    條件：
    1. 收盤 > MA20
    2. RSI(5) 黃金交叉 RSI(10)
    3. 成交量 > VMA5 * 2.0
    """
    if len(df) < 15:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]
    close = last["close"]
    ma20 = last.get("s_ma20")
    rsi5_now = last.get("s_rsi5")
    rsi10_now = last.get("s_rsi10")
    rsi5_prev = prev.get("s_rsi5")
    rsi10_prev = prev.get("s_rsi10")
    volume = last["volume"]
    vol_ma5 = last.get("s_vol_ma5")

    if any(pd.isna(v) for v in [ma20, rsi5_now, rsi10_now, rsi5_prev, rsi10_prev, vol_ma5]):
        return None

    # 1. 位置判定
    if close <= ma20:
        return None

    # 2. RSI 黃金交叉
    if not (rsi5_prev <= rsi10_prev and rsi5_now > rsi10_now):
        return None

    # 3. 攻擊量
    if vol_ma5 == 0 or volume <= vol_ma5 * 2.0:
        return None

    return {
        "strategy": "B",
        "label": "多頭續強",
        "rsi5": round(float(rsi5_now), 2),
        "rsi10": round(float(rsi10_now), 2),
        "vol_ratio": round(float(volume / vol_ma5), 2),
    }


# ═══════════════════════════════════════════════════════════
# 策略 C：W 底反轉型 (Double Bottom Recognition)
# ═══════════════════════════════════════════════════════════

def strategy_c_double_bottom(df: pd.DataFrame, **kwargs) -> dict | None:
    """
    W 底反轉偵測（強化版）：
    1. 過去 60 天用 find_peaks 找低點，prominence 依價格水平自適應
    2. L1 與 L2 價差 < 3%，時間間隔 15-40 天
    3. 今日收盤突破頸線（兩腳間最高點）且 > L1 與 L2 的最高點
    4. MA20 扣抵向上（MA20[-1] > MA20[-5]）
    5. 量能確認：右腳量 < 左腳量（賣壓竭盡），突破日放量 > VMA5 * 1.5
    """
    if len(df) < 65:
        return None

    # 取最近 60 個交易日
    window = df.tail(60).copy()
    close_arr = window["close"].values
    volume_arr = window["volume"].values
    vol_ma5 = df["s_vol_ma5"].iloc[-1]

    if pd.isna(vol_ma5) or vol_ma5 == 0:
        return None

    # 找低點（反轉 close 以尋找谷底）
    mean_price = close_arr.mean()
    daily_range = np.abs(np.diff(close_arr))
    avg_range = daily_range.mean() if len(daily_range) > 0 else mean_price * 0.02
    prominence = max(avg_range * 3, mean_price * 0.01)

    peaks, properties = find_peaks(-close_arr, distance=5, prominence=prominence)

    if len(peaks) < 2:
        return None

    # 取最近兩個低點
    L1_idx = peaks[-2]
    L2_idx = peaks[-1]
    L1_price = close_arr[L1_idx]
    L2_price = close_arr[L2_idx]

    # 對稱性：價差 < 3%
    price_diff = abs(L1_price - L2_price) / L1_price
    if price_diff >= 0.03:
        return None

    # 時間間隔 15-40 天
    gap = L2_idx - L1_idx
    if gap < 15 or gap > 40:
        return None

    # 頸線：兩腳之間的最高點
    neckline = close_arr[L1_idx:L2_idx + 1].max()
    double_bottom_high = max(L1_price, L2_price)

    # 突破判定
    today_close = df["close"].iloc[-1]
    today_volume = df["volume"].iloc[-1]
    if today_close <= neckline or today_close <= double_bottom_high:
        return None

    # MA20 扣抵向上
    if len(df) < 6:
        return None
    ma20_now = df["s_ma20"].iloc[-1]
    ma20_5ago = df["s_ma20"].iloc[-5]
    if pd.isna(ma20_now) or pd.isna(ma20_5ago) or ma20_now <= ma20_5ago:
        return None

    # 量能確認
    l1_vol = volume_arr[max(0, L1_idx - 1):L1_idx + 2].mean()
    l2_vol = volume_arr[max(0, L2_idx - 1):L2_idx + 2].mean()
    vol_exhausted = l2_vol < l1_vol

    if today_volume <= vol_ma5 * 1.5:
        return None

    return {
        "strategy": "C",
        "label": "W底反轉",
        "L1_price": round(float(L1_price), 2),
        "L2_price": round(float(L2_price), 2),
        "neckline": round(float(neckline), 2),
        "gap_days": int(gap),
        "price_diff_pct": round(price_diff * 100, 2),
        "vol_ratio": round(float(today_volume / vol_ma5), 2),
        "ma20_slope": round(float(ma20_now - ma20_5ago), 2),
        "vol_exhausted": vol_exhausted,
    }


# ═══════════════════════════════════════════════════════════
# 策略 D：籌碼同步型 (Smart Money Flow)
# ═══════════════════════════════════════════════════════════

def strategy_d_smart_money(df: pd.DataFrame, inst_df: pd.DataFrame = None, **kwargs) -> dict | None:
    """
    籌碼同步型（強化版）：
    條件：
    1. 符合策略 B 之多頭趨勢條件
    2. 外資或投信連續 n 日買超
    3. 法人合買（外資+投信同時為正）→ 額外加分標記
    4. 若法人資料不可用，改以 MACD 柱狀體連 n 日增長作替代
    """
    n_consecutive = kwargs.get("n_consecutive", 3)
    if inst_df is None:
        inst_df = pd.DataFrame()

    # 先檢查策略 B 條件
    b_result = strategy_b_trend(df)
    if b_result is None:
        return None

    inst_signal = None
    foreign_lots = 0
    trust_lots = 0
    macd_fallback = False
    both_buying = False

    if not inst_df.empty:
        fok, flots = check_consecutive_buying(inst_df, "Foreign_Investor", n_consecutive)
        foreign_lots = flots

        tok, tlots = check_consecutive_buying(inst_df, "Investment_Trust", n_consecutive)
        trust_lots = tlots

        if fok and tok:
            inst_signal = "外資+投信"
            both_buying = True
        elif fok:
            inst_signal = "外資"
            both_buying = check_latest_day_buying(inst_df, "Investment_Trust")
        elif tok:
            inst_signal = "投信"
            both_buying = check_latest_day_buying(inst_df, "Foreign_Investor")
    else:
        # MACD 柱狀體遞增替代
        hist = df["s_macd_hist"].dropna().tail(n_consecutive)
        if len(hist) >= n_consecutive:
            diffs = hist.diff().dropna()
            if len(diffs) >= n_consecutive - 1 and (diffs > 0).all():
                inst_signal = "MACD遞增"
                macd_fallback = True

    if inst_signal is None:
        return None

    label = "籌碼同步（合買）" if both_buying else "籌碼同步"

    return {
        "strategy": "D",
        "label": label,
        "inst_signal": inst_signal,
        "consecutive_days": n_consecutive,
        "foreign_net_lots": foreign_lots,
        "trust_net_lots": trust_lots,
        "both_buying": both_buying,
        "macd_fallback": macd_fallback,
        "vol_ratio": b_result["vol_ratio"],
    }


# ═══════════════════════════════════════════════════════════
# 策略 E：AI 強勢股回測 (AI Sector Pullback) — 池 A
# ═══════════════════════════════════════════════════════════

def strategy_e_pullback(df: pd.DataFrame, industry: str = "", inst_df: pd.DataFrame = None, **kwargs) -> dict | None:
    """
    【池 A】左側支撐觀察池：僅限 AI 相關產業，偵測回測月線支撐的低風險買點。
    """
    if industry not in AI_INDUSTRIES:
        return None

    if len(df) < 30:
        return None

    last = df.iloc[-1]
    close = last["close"]
    ma20 = last.get("s_ma20")
    rsi5 = last.get("s_rsi5")
    volume = last["volume"]
    vol_ma5 = last.get("s_vol_ma5")

    if any(pd.isna(v) for v in [ma20, rsi5, vol_ma5]) or ma20 == 0 or vol_ma5 == 0:
        return None

    # 1. MA20 扣抵向上
    if len(df) < 6:
        return None
    ma20_now = df["s_ma20"].iloc[-1]
    ma20_5ago = df["s_ma20"].iloc[-5]
    if pd.isna(ma20_5ago) or ma20_now <= ma20_5ago:
        return None

    # 2. 回測支撐：收盤在 MA20 附近（-3% ~ +8%）
    dist_pct = (close - ma20) / ma20
    if dist_pct < -0.03 or dist_pct > 0.08:
        return None

    # 3. 動能降溫：RSI(5) < 55
    if rsi5 >= 55:
        return None

    # 4. 量能枯竭：今日量 < VMA5 × 0.8
    if volume >= vol_ma5 * 0.8:
        return None

    # 5. 法人加分
    trust_buying = False
    if inst_df is not None and not inst_df.empty:
        trust_buying = check_latest_day_buying(inst_df, "Investment_Trust")

    label = "池A：回測洗盤" if trust_buying else "池A：回測支撐"

    return {
        "strategy": "E",
        "label": label,
        "pool": "A",
        "industry": industry,
        "ma20": round(float(ma20), 2),
        "ma20_slope": round(float(ma20_now - ma20_5ago), 2),
        "dist_ma20_pct": round(dist_pct * 100, 2),
        "rsi5": round(float(rsi5), 2),
        "vol_shrink": round(float(volume / vol_ma5), 2),
        "trust_buying": trust_buying,
    }


# ═══════════════════════════════════════════════════════════
# 策略 F：【池 B】右側動能確認（朱家泓法）
# ═══════════════════════════════════════════════════════════

def strategy_f_momentum(df: pd.DataFrame, industry: str = "", **kwargs) -> dict | None:
    """
    【池 B】右側動能確認（朱家泓法）：捕捉多頭結構最完美的攻擊標的。
    """
    if industry not in AI_INDUSTRIES:
        return None

    if len(df) < 65:
        return None

    last = df.iloc[-1]
    close = float(last["close"])
    ma5 = last.get("s_ma5")
    ma10 = last.get("s_ma10")
    ma20 = last.get("s_ma20")
    ma60 = last.get("s_ma60")
    volume = float(last["volume"])
    vol_ma5 = last.get("s_vol_ma5")

    if any(pd.isna(v) for v in [ma5, ma10, ma20, ma60, vol_ma5]):
        return None
    ma5, ma10, ma20, ma60, vol_ma5 = (
        float(ma5), float(ma10), float(ma20), float(ma60), float(vol_ma5),
    )
    if ma60 == 0 or vol_ma5 == 0:
        return None

    # 1. 型態：頭頭高底底高
    swing = detect_swing_highs_lows(df, lookback=40)
    if swing is None:
        return None

    if swing["latest_swing_high"] <= swing["prev_swing_high"]:
        return None
    if swing["latest_swing_low"] <= swing["prev_swing_low"]:
        return None

    # 突破前 20 日最高
    rolling_high_20 = float(df["close"].rolling(20).max().iloc[-1])
    if close < rolling_high_20:
        return None

    # 2. 四線多排
    if not (ma5 > ma10 > ma20 > ma60):
        return None
    if close <= ma20 or close <= ma60:
        return None

    # 3. 進場確認
    if close <= ma5:
        return None
    if volume <= vol_ma5 * 1.5:
        return None

    dist_ma20_pct = (close - ma20) / ma20

    return {
        "strategy": "F",
        "label": "池B：動能確認",
        "pool": "B",
        "industry": industry,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "dist_ma20_pct": round(dist_ma20_pct * 100, 2),
        "vol_ratio": round(volume / vol_ma5, 2),
        "swing_high_prev": round(swing["prev_swing_high"], 2),
        "swing_high_latest": round(swing["latest_swing_high"], 2),
        "swing_low_prev": round(swing["prev_swing_low"], 2),
        "swing_low_latest": round(swing["latest_swing_low"], 2),
        "rolling_high_20": round(rolling_high_20, 2),
    }


# ═══════════════════════════════════════════════════════════
# 策略註冊
# ═══════════════════════════════════════════════════════════

register(StrategyInfo(
    code="A", name="均線糾結起漲",
    description="MA5/10/20 糾結 + 量增突破",
    category="classic", pool=None,
    needs_institutional=False, needs_industry=True,
    css_class="strat-a", func=strategy_a_spring,
))

register(StrategyInfo(
    code="B", name="多頭續強",
    description="RSI5 黃金交叉 + 強勢量能",
    category="classic", pool=None,
    needs_institutional=False, needs_industry=False,
    css_class="strat-b", func=strategy_b_trend,
))

register(StrategyInfo(
    code="C", name="W底反轉",
    description="雙底型態 + 頸線突破 + 放量確認",
    category="classic", pool=None,
    needs_institutional=False, needs_industry=False,
    css_class="strat-c", func=strategy_c_double_bottom,
))

register(StrategyInfo(
    code="D", name="籌碼同步",
    description="多頭趨勢 + 法人連續買超",
    category="classic", pool=None,
    needs_institutional=True, needs_industry=False,
    css_class="strat-d", func=strategy_d_smart_money,
))

register(StrategyInfo(
    code="E", name="池A：回測支撐",
    description="AI 產業回測月線 + 量縮 + RSI降溫",
    category="classic", pool="A",
    needs_institutional=True, needs_industry=True,
    css_class="strat-e", func=strategy_e_pullback,
))

register(StrategyInfo(
    code="F", name="池B：動能確認",
    description="頭頭高底底高 + 四線多排 + 帶量攻擊",
    category="classic", pool="B",
    needs_institutional=False, needs_industry=True,
    css_class="strat-f", func=strategy_f_momentum,
))
