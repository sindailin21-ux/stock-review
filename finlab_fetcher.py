"""
finlab_fetcher.py
FinLab 資料層 + 每日磁碟快取 + 技術指標預計算。

一次呼叫取全市場資料（~2,700 檔），快取後同日重複掃描 0 API 呼叫。
技術指標（MA, RSI, ADX, MACD 等）使用 FinLab data.indicator()（TA-Lib 引擎）。
"""
from __future__ import annotations

import os
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# ── 模組級快取 ──
_cache: dict = {}
_cache_date: str | None = None
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")


# ═══════════════════════════════════════════════════════════
# FinLab 登入
# ═══════════════════════════════════════════════════════════

def login():
    """
    FinLab 登入。
    優先使用 FINLAB_API_TOKEN 環境變數；若未設定則嘗試快取憑證。
    """
    import finlab
    from config import FINLAB_API_TOKEN
    token = FINLAB_API_TOKEN or ""
    finlab.login(token)


# ═══════════════════════════════════════════════════════════
# 每日快取：載入 / 建立
# ═══════════════════════════════════════════════════════════

def load_daily_cache(target_date: str | None = None, status: dict | None = None):
    """
    載入或建立每日快取（含 FinLab 預計算指標）。
    target_date: "YYYY-MM-DD"，預設今日。
    status: 可選，用於回報進度（{"current": "..."} 格式）。
    """
    global _cache, _cache_date
    from finlab import data

    date_str = target_date or datetime.now().strftime("%Y-%m-%d")

    # 記憶體快取命中
    if _cache_date == date_str and _cache:
        if status:
            status["current"] = f"⚡ 記憶體快取命中（{date_str}）"
        print(f"⚡ FinLab 記憶體快取命中：{date_str}")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    pkl_path = os.path.join(CACHE_DIR, f"finlab_{date_str}.pkl")

    # 磁碟快取命中
    if os.path.exists(pkl_path):
        if status:
            status["current"] = f"⚡ 載入 FinLab 磁碟快取（{date_str}）..."
        print(f"⚡ 載入 FinLab 磁碟快取：{pkl_path}")
        with open(pkl_path, "rb") as f:
            _cache = pickle.load(f)
        _cache_date = date_str
        return

    # ── 從 FinLab 批次抓取全市場資料 ──
    login()
    _cache = {"date": date_str}

    # 記憶體優化：只保留最近 N 天 + 轉 float32
    # 選股只需 ~250 天（5 個月 × 30 + 100 暖機），400 天足夠
    _TRIM_ROWS = 400

    def _trim(df):
        """裁剪到最近 N 行 + 轉 float32，大幅節省記憶體。"""
        if df is None or not hasattr(df, "iloc"):
            return df
        trimmed = df.iloc[-_TRIM_ROWS:].copy() if len(df) > _TRIM_ROWS else df.copy()
        # float64 → float32 省一半記憶體
        float64_cols = trimmed.select_dtypes(include=[np.float64]).columns
        if len(float64_cols):
            trimmed[float64_cols] = trimmed[float64_cols].astype(np.float32)
        return trimmed

    # 【價量】
    if status:
        status["current"] = "FinLab：載入全市場價量資料..."
    print("📥 FinLab：載入全市場價量...")
    _cache["close"]  = _trim(data.get("price:收盤價"))
    _cache["open"]   = _trim(data.get("price:開盤價"))
    _cache["high"]   = _trim(data.get("price:最高價"))
    _cache["low"]    = _trim(data.get("price:最低價"))
    _cache["volume"] = _trim(data.get("price:成交股數"))

    # 【技術指標 — FinLab data.indicator()，TA-Lib 引擎】
    if status:
        status["current"] = "FinLab：計算全市場技術指標..."
    print("📥 FinLab：計算全市場技術指標...")
    _cache["sma5"]  = _trim(data.indicator("SMA", timeperiod=5))
    _cache["sma10"] = _trim(data.indicator("SMA", timeperiod=10))
    _cache["sma20"] = _trim(data.indicator("SMA", timeperiod=20))
    _cache["sma60"] = _trim(data.indicator("SMA", timeperiod=60))

    _cache["rsi5"]  = _trim(data.indicator("RSI", timeperiod=5))
    _cache["rsi10"] = _trim(data.indicator("RSI", timeperiod=10))
    _cache["rsi14"] = _trim(data.indicator("RSI", timeperiod=14))

    _cache["adx8"]      = _trim(data.indicator("ADX", timeperiod=8))
    _cache["plus_di8"]  = _trim(data.indicator("PLUS_DI", timeperiod=8))
    _cache["minus_di8"] = _trim(data.indicator("MINUS_DI", timeperiod=8))

    # MACD 回傳 tuple: (macd, signal, hist)
    macd_result = data.indicator("MACD", fastperiod=6, slowperiod=13, signalperiod=9)
    _cache["macd"]        = _trim(macd_result[0])
    _cache["macd_signal"] = _trim(macd_result[1])
    _cache["macd_hist"]   = _trim(macd_result[2])

    # 【法人買賣超】
    if status:
        status["current"] = "FinLab：載入法人買賣超..."
    print("📥 FinLab：載入法人買賣超...")
    _cache["foreign_net"] = _trim(data.get(
        "institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)"
    ))
    _cache["trust_net"] = _trim(data.get(
        "institutional_investors_trading_summary:投信買賣超股數"
    ))

    # 【營收】— 月頻資料不裁剪行數，但轉 float32
    if status:
        status["current"] = "FinLab：載入營收資料..."
    print("📥 FinLab：載入營收資料...")
    _cache["revenue"]      = _trim(data.get("monthly_revenue:當月營收"))
    _cache["prev_revenue"] = _trim(data.get("monthly_revenue:上月營收"))
    _cache["yoy_revenue"]  = _trim(data.get("monthly_revenue:去年當月營收"))

    # 【EPS】— 季頻資料不裁剪
    if status:
        status["current"] = "FinLab：載入 EPS 資料..."
    print("📥 FinLab：載入 EPS...")
    _cache["eps"] = _trim(data.get("financial_statement:每股盈餘"))

    # 計算記憶體用量
    total_mb = sum(
        v.memory_usage(deep=True).sum() for v in _cache.values()
        if hasattr(v, "memory_usage")
    ) / 1024 / 1024
    print(f"📊 FinLab 快取記憶體用量：{total_mb:.0f} MB")

    # 存磁碟快取
    if status:
        status["current"] = "FinLab：儲存磁碟快取..."
    print(f"💾 FinLab 磁碟快取儲存中：{pkl_path}")
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(_cache, f)
        pkl_size = os.path.getsize(pkl_path) / 1024 / 1024
        print(f"✅ FinLab 快取已儲存（{pkl_size:.0f} MB）")
    except Exception as e:
        print(f"⚠️ FinLab 快取儲存失敗：{e}")

    _cache_date = date_str


# ═══════════════════════════════════════════════════════════
# 核心：取得單股 enriched DataFrame（OHLCV + s_* 指標）
# ═══════════════════════════════════════════════════════════

# 指標映射表：cache key → 策略使用的欄位名
_INDICATOR_MAP = {
    "sma5":      "s_ma5",
    "sma10":     "s_ma10",
    "sma20":     "s_ma20",
    "sma60":     "s_ma60",
    "rsi5":      "s_rsi5",
    "rsi10":     "s_rsi10",
    "rsi14":     "s_rsi14",
    "adx8":      "s_adx14",       # ADX(8)，欄位名保持 s_adx14 向下相容
    "plus_di8":  "s_plus_di14",   # +DI(8)
    "minus_di8": "s_minus_di14",  # -DI(8)
    "macd":      "s_macd",
    "macd_signal": "s_macd_signal",
    "macd_hist": "s_macd_hist",
}


def get_enriched_df(stock_id: str, months: int = 5) -> pd.DataFrame:
    """
    從快取組裝單股 DataFrame，含 OHLCV + 全部 s_* 指標欄位。
    回傳格式與 compute_screener_indicators() 輸出完全相同，策略可直接使用。
    """
    if not _cache:
        return pd.DataFrame()

    sid = str(stock_id)
    close_wide = _cache.get("close")
    if close_wide is None or sid not in close_wide.columns:
        return pd.DataFrame()

    # 多取 100 天給長週期指標暖機
    n_rows = months * 30 + 100

    # 取可用範圍
    total_rows = len(close_wide)
    n_rows = min(n_rows, total_rows)

    # 建立日期 index
    dates = close_wide.index[-n_rows:]

    # 組裝 OHLCV — 先取各欄位 Series
    price_cols = {}
    for key, col_name in [("open", "open"), ("high", "high"),
                           ("low", "low"), ("close", "close"),
                           ("volume", "volume")]:
        wide = _cache.get(key)
        if wide is not None and sid in wide.columns:
            price_cols[col_name] = wide[sid].iloc[-n_rows:].values
        else:
            return pd.DataFrame()  # 缺少必要欄位

    df = pd.DataFrame(price_cols, index=dates)
    df.index.name = "date"
    df = df.reset_index()

    # 移除 close 為 NaN 的列（非交易日）
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    if df.empty or len(df) < 20:
        return df

    # ── 映射 FinLab 指標到 s_* 欄位 ──
    # 需要重新對齊：先建立日期索引，再 merge
    df_dates = pd.to_datetime(df["date"])

    for cache_key, col_name in _INDICATOR_MAP.items():
        wide = _cache.get(cache_key)
        if wide is not None and sid in wide.columns:
            indicator_series = wide[sid]
            # 用日期對齊
            aligned = indicator_series.reindex(pd.to_datetime(df["date"]))
            df[col_name] = aligned.values

    # ── vol_ma5 / vol_ratio 本地計算 ──
    df["s_vol_ma5"] = df["volume"].rolling(window=5).mean().round(0)
    vol_ma5 = df["s_vol_ma5"].replace(0, np.nan)
    df["s_vol_ratio"] = (df["volume"] / vol_ma5).round(2)

    return df


def get_price_df(stock_id: str, months: int = 5) -> pd.DataFrame:
    """
    從快取取純 OHLCV DataFrame（不含 s_* 指標）。
    用於需要合併即時報價後重算指標的場景（朱家泓覆盤 / H策略診斷）。
    """
    if not _cache:
        return pd.DataFrame()

    sid = str(stock_id)
    close_wide = _cache.get("close")
    if close_wide is None or sid not in close_wide.columns:
        return pd.DataFrame()

    n_rows = months * 30 + 100
    total_rows = len(close_wide)
    n_rows = min(n_rows, total_rows)

    dates = close_wide.index[-n_rows:]
    price_cols = {}
    for key, col_name in [("open", "open"), ("high", "high"),
                           ("low", "low"), ("close", "close"),
                           ("volume", "volume")]:
        wide = _cache.get(key)
        if wide is not None and sid in wide.columns:
            price_cols[col_name] = wide[sid].iloc[-n_rows:].values
        else:
            return pd.DataFrame()

    df = pd.DataFrame(price_cols, index=dates)
    df.index.name = "date"
    df = df.reset_index()
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════
# 法人買賣超
# ═══════════════════════════════════════════════════════════

def get_institutional_df(stock_id: str, days: int = 20) -> pd.DataFrame:
    """
    回傳法人買賣超 DataFrame，格式相容現有策略。
    欄位：date, Foreign_Investor_Buy, Foreign_Investor_Sell,
          Investment_Trust_Buy, Investment_Trust_Sell
    FinLab 提供淨買超（正=買超，負=賣超），需轉換為 Buy/Sell 格式。
    """
    if not _cache:
        return pd.DataFrame()

    sid = str(stock_id)
    foreign_wide = _cache.get("foreign_net")
    trust_wide = _cache.get("trust_net")

    if foreign_wide is None or sid not in foreign_wide.columns:
        return pd.DataFrame()

    # 取近 N 天
    foreign = foreign_wide[sid].tail(days).copy()
    trust = trust_wide[sid].tail(days).copy() if (
        trust_wide is not None and sid in trust_wide.columns
    ) else pd.Series(0, index=foreign.index)

    # 轉換淨買超 → Buy/Sell 格式
    # 正值 → Buy=net, Sell=0 ; 負值 → Buy=0, Sell=abs(net)
    df = pd.DataFrame({"date": foreign.index})
    df["date"] = pd.to_datetime(df["date"])

    foreign_vals = foreign.values
    trust_vals = trust.values

    df["Foreign_Investor_Buy"]  = np.where(foreign_vals > 0, foreign_vals, 0).astype(float)
    df["Foreign_Investor_Sell"] = np.where(foreign_vals < 0, -foreign_vals, 0).astype(float)
    df["Investment_Trust_Buy"]  = np.where(trust_vals > 0, trust_vals, 0).astype(float)
    df["Investment_Trust_Sell"] = np.where(trust_vals < 0, -trust_vals, 0).astype(float)

    df = df.dropna(subset=["Foreign_Investor_Buy"]).reset_index(drop=True)
    return df


def get_latest_institutional_net(stock_id: str) -> tuple:
    """
    回傳最新一日法人淨買超（張）。
    回傳 (foreign_net_lots: int|None, trust_net_lots: int|None)
    """
    if not _cache:
        return (None, None)

    sid = str(stock_id)
    foreign_wide = _cache.get("foreign_net")
    trust_wide = _cache.get("trust_net")

    foreign_net = None
    trust_net = None

    if foreign_wide is not None and sid in foreign_wide.columns:
        vals = foreign_wide[sid].dropna()
        if not vals.empty:
            foreign_net = int(vals.iloc[-1]) // 1000  # 股→張

    if trust_wide is not None and sid in trust_wide.columns:
        vals = trust_wide[sid].dropna()
        if not vals.empty:
            trust_net = int(vals.iloc[-1]) // 1000  # 股→張

    return (foreign_net, trust_net)


# ═══════════════════════════════════════════════════════════
# 營收
# ═══════════════════════════════════════════════════════════

def get_revenue_df(stock_id: str) -> pd.DataFrame:
    """
    回傳營收 DataFrame，格式相容 get_revenue_summary()。
    欄位：date, revenue, revenue_month, revenue_year
    """
    if not _cache:
        return pd.DataFrame()

    sid = str(stock_id)
    rev_wide = _cache.get("revenue")
    if rev_wide is None or sid not in rev_wide.columns:
        return pd.DataFrame()

    # 取近 24 個月
    rev = rev_wide[sid].dropna().tail(24)
    if rev.empty:
        return pd.DataFrame()

    yoy_wide = _cache.get("yoy_revenue")

    df = pd.DataFrame({
        "date": rev.index,
        "revenue": rev.values,
    })
    df["date"] = pd.to_datetime(df["date"])

    # 從日期提取 year/month（供 get_revenue_summary 比對用）
    df["revenue_year"] = df["date"].dt.year
    df["revenue_month"] = df["date"].dt.month

    # FinLab 營收單位是「千元」，轉成「元」以相容 get_revenue_summary
    df["revenue"] = df["revenue"] * 1000

    return df.reset_index(drop=True)


def get_revenue_yoy(stock_id: str):
    """
    直接回傳最新月營收年增率 (%)，不走 get_revenue_summary。
    用 FinLab 的「當月營收」和「去年當月營收」直接計算。
    """
    if not _cache:
        return None

    sid = str(stock_id)
    rev_wide = _cache.get("revenue")
    yoy_wide = _cache.get("yoy_revenue")

    if rev_wide is None or sid not in rev_wide.columns:
        return None
    if yoy_wide is None or sid not in yoy_wide.columns:
        return None

    # 取最新一筆非 NaN
    rev = rev_wide[sid].dropna()
    yoy_rev = yoy_wide[sid].dropna()
    if rev.empty or yoy_rev.empty:
        return None

    # 取最新月份（共同最新日期）
    latest_date = rev.index[-1]
    current = rev.iloc[-1]

    # 找對應的去年同月營收
    if latest_date in yoy_rev.index:
        prev_year = yoy_rev.loc[latest_date]
    else:
        prev_year = yoy_rev.iloc[-1]

    if prev_year and prev_year > 0:
        yoy = (current - prev_year) / prev_year * 100
        if abs(yoy) <= 1000:  # 排除異常值
            return round(yoy, 2)

    return None


# ═══════════════════════════════════════════════════════════
# EPS
# ═══════════════════════════════════════════════════════════

def get_eps(stock_id: str) -> tuple:
    """
    回傳 (latest_eps: float, quarter_str: str, eps_list: list[float])。
    eps_list 為近 8 季 EPS（由舊到新）。
    若無資料回傳 (None, None, [])。
    """
    if not _cache:
        return (None, None, [])

    sid = str(stock_id)
    eps_wide = _cache.get("eps")
    if eps_wide is None or sid not in eps_wide.columns:
        return (None, None, [])

    eps_series = eps_wide[sid].dropna().tail(8)
    if eps_series.empty:
        return (None, None, [])

    latest_eps = float(eps_series.iloc[-1])
    latest_date = eps_series.index[-1]
    quarter_str = str(latest_date)[:7]  # "2025-Q3" 或 "2025-09"
    eps_list = eps_series.values.tolist()

    return (latest_eps, quarter_str, eps_list)


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def get_all_stock_ids() -> list:
    """回傳快取中所有股票代號（用於全市場掃描）。"""
    if not _cache:
        return []
    close_wide = _cache.get("close")
    if close_wide is None:
        return []
    return list(close_wide.columns)


def reset_cache():
    """清除記憶體快取（換日時呼叫）。"""
    global _cache, _cache_date
    _cache = {}
    _cache_date = None


def get_cache_info() -> dict:
    """回傳快取資訊（供 UI 顯示）。"""
    return {
        "date": _cache_date,
        "loaded": bool(_cache),
        "n_stocks": len(_cache.get("close", {}).columns) if _cache.get("close") is not None else 0,
    }
