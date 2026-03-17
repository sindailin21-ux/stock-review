"""
strategies/_helpers.py
策略共用工具函式與常數。
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks


# ── 產業分類常數 ──

HIGH_VOL_INDUSTRIES = {"半導體業", "電腦及週邊設備業", "電子零組件業"}

AI_CORE_INDUSTRIES = {
    "半導體業",
    "電腦及週邊設備業",
    "電子零組件業",
}

AI_INDUSTRIES = AI_CORE_INDUSTRIES | {
    "光電業",
    "通信網路業",
    "電子通路業",
    "資訊服務業",
    "其他電子業",
}


# ── 法人資料工具 ──

def check_consecutive_buying(inst_df: pd.DataFrame, investor_type: str, n: int = 3) -> tuple:
    """
    檢查某法人類型是否連續 n 日買超。
    回傳 (is_consecutive: bool, latest_net_lots: int)
    """
    if inst_df.empty:
        return False, 0

    # 長格式：name / buy / sell
    if "name" in inst_df.columns:
        sub = inst_df[inst_df["name"] == investor_type].copy()
        if sub.empty or "buy" not in sub.columns:
            return False, 0
        sub = sub.sort_values("date").tail(n)
        if len(sub) < n:
            return False, 0
        sub["net"] = sub["buy"] - sub["sell"]
        if (sub["net"] > 0).all():
            latest_lots = int(sub["net"].iloc[-1] // 1000)
            return True, latest_lots
        return False, 0

    # 寬格式：Foreign_Investor_Buy / Foreign_Investor_Sell
    buy_col = f"{investor_type}_Buy"
    sell_col = f"{investor_type}_Sell"
    if buy_col in inst_df.columns and sell_col in inst_df.columns:
        sub = inst_df.sort_values("date").tail(n)
        if len(sub) < n:
            return False, 0
        net = sub[buy_col] - sub[sell_col]
        if (net > 0).all():
            latest_lots = int(net.iloc[-1] // 1000)
            return True, latest_lots

    return False, 0


def check_latest_day_buying(inst_df: pd.DataFrame, investor_type: str) -> bool:
    """
    檢查某法人最新一天是否淨買超（非連買，只看最新 1 天）。
    用於「法人合買」加分判定。
    """
    if inst_df.empty:
        return False

    if "name" in inst_df.columns:
        sub = inst_df[inst_df["name"] == investor_type].copy()
        if sub.empty or "buy" not in sub.columns:
            return False
        latest = sub.sort_values("date").iloc[-1]
        return (latest["buy"] - latest["sell"]) > 0

    buy_col = f"{investor_type}_Buy"
    sell_col = f"{investor_type}_Sell"
    if buy_col in inst_df.columns and sell_col in inst_df.columns:
        latest = inst_df.sort_values("date").iloc[-1]
        return (latest[buy_col] - latest[sell_col]) > 0

    return False


def check_distribution_top(df: pd.DataFrame, foreign_net: int | None, trust_net: int | None,
                           prev_foreign_net: int | None = None, prev_trust_net: int | None = None) -> bool:
    """
    高檔出貨過濾（兩種模式，任一成立即排除）：

    模式 A — 高檔出貨：長黑K + 位階>70% + 法人賣超
    模式 B — 誘多翻臉：長黑K + 法人前日買超→今日賣超且賣超量≥前日買超量（不限位階）

    三個/兩個條件全部成立才回傳 True。
    """
    if len(df) < 60:
        return False

    last = df.iloc[-1]
    close = float(last["close"])
    open_ = float(last["open"])

    # 共用條件：長黑K（收黑 且 實體 > 3%）
    if close >= open_:
        return False
    body_pct = (open_ - close) / open_ * 100
    if body_pct <= 3:
        return False

    # ── 模式 A：高檔出貨（位階>70% + 法人賣超）──
    window = df.tail(60)
    high_60 = float(window["high"].max())
    low_60 = float(window["low"].min())
    if high_60 != low_60:
        position = (close - low_60) / (high_60 - low_60) * 100
        if position > 70:
            foreign_selling = foreign_net is not None and foreign_net < 0
            trust_selling = trust_net is not None and trust_net < 0
            if foreign_selling or trust_selling:
                return True

    # ── 模式 B：誘多翻臉（前日買超 → 今日賣超 ≥ 前日買超量）──
    # 外資：前日買超 > 0，今日賣超且 |賣超| ≥ 前日買超
    if (prev_foreign_net is not None and prev_foreign_net > 0
            and foreign_net is not None and foreign_net < 0
            and abs(foreign_net) >= prev_foreign_net):
        return True

    # 投信：前日買超 > 0，今日賣超且 |賣超| ≥ 前日買超
    if (prev_trust_net is not None and prev_trust_net > 0
            and trust_net is not None and trust_net < 0
            and abs(trust_net) >= prev_trust_net):
        return True

    return False


def detect_swing_highs_lows(df: pd.DataFrame, lookback: int = 40) -> dict | None:
    """
    偵測近 `lookback` 根 K 線的擺盪高低點，用於判斷「頭頭高底底高」型態。
    回傳 dict: prev_swing_high, latest_swing_high, prev_swing_low, latest_swing_low
    或 None（資料不足/型態不成立）。
    """
    if len(df) < lookback:
        return None

    window = df.tail(lookback)
    close_arr = window["close"].values.astype(float)

    # 動態 prominence：均價的 2%
    avg_price = close_arr.mean()
    prominence = max(avg_price * 0.02, 0.5)

    # 擺盪高點（peaks）
    highs, _ = find_peaks(close_arr, distance=5, prominence=prominence)
    # 擺盪低點（peaks of inverted series）
    lows, _ = find_peaks(-close_arr, distance=5, prominence=prominence)

    if len(highs) < 2 or len(lows) < 2:
        return None

    return {
        "prev_swing_high": float(close_arr[highs[-2]]),
        "latest_swing_high": float(close_arr[highs[-1]]),
        "prev_swing_low": float(close_arr[lows[-2]]),
        "latest_swing_low": float(close_arr[lows[-1]]),
    }


def check_profitability(stock_id: str) -> tuple:
    """
    精煉型基本面濾網（全策略通用）。

    條件 A（主力門檻）：最新季 EPS > 0 且 營收 YoY > config 門檻（預設 0%）。
    條件 B（轉機門檻）：最新季 EPS < 0，但同時滿足：
       B1. 虧損收斂：本季 EPS > 上季 EPS
       B2. 營收 YoY > 40%
       → 通過，標記「轉機潛力股」

    營收 YoY 門檻由 config.PROFITABILITY_REV_YOY_THRESHOLD 控制（%）。
    回測結論：0% 門檻（任何正成長）在風險調整後最優。

    回傳 (pass: bool, reason: str | None)

    優先使用 FinLab 快取；若 FinLab 快取不可用則 fallback 到 FinMind。
    """
    # ── 嘗試從 FinLab 快取取得 ──
    rev_yoy = None
    latest_eps = None
    prev_eps = None
    use_finlab = False

    try:
        import finlab_fetcher
        if finlab_fetcher._cache:
            # EPS
            eps_result = finlab_fetcher.get_eps(stock_id)
            if eps_result[0] is not None:
                latest_eps = eps_result[0]
                eps_list = eps_result[2]
                if len(eps_list) >= 2:
                    prev_eps = eps_list[-2]
                use_finlab = True

            # 營收 YoY
            yoy_val = finlab_fetcher.get_revenue_yoy(stock_id)
            if yoy_val is not None:
                rev_yoy = yoy_val
    except Exception:
        pass

    # ── FinLab 快取不可用 → fallback 到 FinMind ──
    if not use_finlab:
        from data_fetcher import _fetch, fetch_revenue
        from fundamentals import get_revenue_summary

        # 營收年增率
        try:
            rev_df = fetch_revenue(stock_id)
            rev_summary = get_revenue_summary(rev_df)
            rev_yoy = rev_summary.get("年增率(%)")
        except Exception:
            pass

        # EPS
        start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
        try:
            df_fin = _fetch("TaiwanStockFinancialStatements", stock_id, start_date=start)
        except Exception:
            return (True, "無法取得財報（預設通過）")

        if df_fin.empty or "type" not in df_fin.columns:
            return (True, "無財報資料（預設通過）")

        eps_df = df_fin[df_fin["type"] == "EPS"].copy()
        if eps_df.empty:
            return (True, "無 EPS 資料（預設通過）")

        eps_df["date"] = pd.to_datetime(eps_df["date"])
        eps_df = eps_df.sort_values("date")
        latest_eps = float(eps_df.iloc[-1]["value"])
        if len(eps_df) >= 2:
            prev_eps = float(eps_df.iloc[-2]["value"])

    # ── 共用判斷邏輯 ──
    if latest_eps is None:
        return (True, "無 EPS 資料（預設通過）")

    # ════ 條件 A：主力門檻 — EPS > 0 且 營收 YoY > 門檻 ════
    from config import PROFITABILITY_REV_YOY_THRESHOLD
    yoy_threshold = PROFITABILITY_REV_YOY_THRESHOLD  # 預設 0（%）
    if latest_eps > 0:
        if rev_yoy is None:
            return (True, "獲利股")
        if rev_yoy > yoy_threshold:
            return (True, "獲利股")
        return (False, f"EPS>0但營收YoY={rev_yoy:+.1f}%未達{yoy_threshold}%")

    # ════ 條件 B：轉機門檻 — EPS ≤ 0 + 虧損收斂 + 營收 YoY > 40% ════
    if prev_eps is None:
        return (False, "EPS≤0 且歷史資料不足")

    # B1. 虧損收斂：本季 EPS > 上季 EPS
    if latest_eps <= prev_eps:
        return (False, f"EPS≤0 且虧損未收斂（{prev_eps:.2f}→{latest_eps:.2f}）")

    # B2. 營收 YoY > 40%
    if rev_yoy is not None and rev_yoy > 40:
        return (True, "轉機潛力股")

    yoy_str = f"{rev_yoy:+.1f}%" if rev_yoy is not None else "無資料"
    return (False, f"EPS 虧損收斂但營收YoY={yoy_str}未達40%")
