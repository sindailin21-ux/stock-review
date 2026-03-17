"""
screener.py
選股掃描引擎 — 指標計算 + 主掃描迴圈。

策略邏輯已搬至 strategies/ 套件，本檔負責：
1. compute_screener_indicators() — 計算技術指標
2. scan_stocks() — 主掃描迴圈（透過 registry 分派策略）
3. estimate_intraday_volume() — 盤中量能估算
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime

from config import MACD_FAST, MACD_SLOW, MACD_SIGNAL
from strategies import discover_strategies, get_strategy
from strategies._helpers import check_profitability

# 初始化策略註冊表
discover_strategies()


# ═══════════════════════════════════════════════════════════
# 指標計算（獨立於 indicators.py，避免互相影響）
# ═══════════════════════════════════════════════════════════

def compute_screener_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算選股策略所需的技術指標。
    MA(5,20,60)、MACD(6,13,9)、RSI(5)、RSI(10)、VMA5

    若 DataFrame 已含 FinLab 預計算指標（s_ma5 欄位存在），直接返回。
    """
    if df.empty or len(df) < 20:
        return df

    # FinLab 預計算指標已存在 → 跳過重複計算
    if "s_ma5" in df.columns:
        return df

    df = df.copy()

    # ── 均線 ──
    for p in [5, 10, 20, 60]:
        df[f"s_ma{p}"] = df["close"].rolling(window=p).mean().round(2)

    # ── MACD (6, 13, 9) ──
    ema_fast = df["close"].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df["close"].ewm(span=MACD_SLOW, adjust=False).mean()
    df["s_macd"] = (ema_fast - ema_slow).round(4)
    df["s_macd_signal"] = df["s_macd"].ewm(span=MACD_SIGNAL, adjust=False).mean().round(4)
    df["s_macd_hist"] = (df["s_macd"] - df["s_macd_signal"]).round(4)

    # ── RSI(5) 與 RSI(10)  ── Wilder's RSI（SMMA / EMA）
    for period in [5, 10]:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        # Wilder's smoothing = EMA with alpha = 1/period
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        # avg_loss == 0 → 完全無跌幅 → RSI = 100
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        rsi = rsi.where(avg_loss > 0, 100.0)   # avg_loss=0 直接設 100
        df[f"s_rsi{period}"] = rsi.round(2)

    # ── RSI(14) ──
    delta14 = df["close"].diff()
    gain14 = delta14.where(delta14 > 0, 0.0)
    loss14 = -delta14.where(delta14 < 0, 0.0)
    avg_gain14 = gain14.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss14 = loss14.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs14 = avg_gain14 / avg_loss14
    rsi14 = 100 - 100 / (1 + rs14)
    rsi14 = rsi14.where(avg_loss14 > 0, 100.0)
    df["s_rsi14"] = rsi14.round(2)

    # ── ADX(8) 趨勢強度（回測最佳天期）──
    if "high" in df.columns and "low" in df.columns:
        import numpy as _np
        _adx_p = 8  # 回測最佳天期
        _high = df["high"]
        _low = df["low"]
        _close = df["close"]

        _tr1 = _high - _low
        _tr2 = (_high - _close.shift(1)).abs()
        _tr3 = (_low - _close.shift(1)).abs()
        _true_range = pd.concat([_tr1, _tr2, _tr3], axis=1).max(axis=1)
        # Wilder's smoothing (EWM alpha=1/period) 與 TA-Lib 一致
        _atr = _true_range.ewm(alpha=1.0 / _adx_p, min_periods=_adx_p, adjust=False).mean()

        _up_move = _high - _high.shift(1)
        _down_move = _low.shift(1) - _low
        _plus_dm = _up_move.where((_up_move > _down_move) & (_up_move > 0), 0.0)
        _minus_dm = _down_move.where((_down_move > _up_move) & (_down_move > 0), 0.0)

        _plus_di = 100 * (_plus_dm.ewm(alpha=1.0 / _adx_p, min_periods=_adx_p, adjust=False).mean() / _atr)
        _minus_di = 100 * (_minus_dm.ewm(alpha=1.0 / _adx_p, min_periods=_adx_p, adjust=False).mean() / _atr)
        _dx = (_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, _np.nan) * 100
        _adx = _dx.ewm(alpha=1.0 / _adx_p, min_periods=_adx_p, adjust=False).mean()

        # 欄位名保持 s_adx14 等（向下兼容策略函式）
        df["s_adx14"] = _adx.round(2)
        df["s_plus_di14"] = _plus_di.round(2)
        df["s_minus_di14"] = _minus_di.round(2)

    # ── 量能 ──
    df["s_vol_ma5"] = df["volume"].rolling(window=5).mean().round(0)
    df["s_vol_ratio"] = (df["volume"] / df["s_vol_ma5"]).round(2)

    return df


# ═══════════════════════════════════════════════════════════
# 盤中量能估算
# ═══════════════════════════════════════════════════════════

def estimate_intraday_volume(
    current_volume: int,
    market_open: datetime,
    now: datetime,
    prev_close_volume: int,
    alert_threshold: float = 2.0,
) -> dict:
    """
    預估量 = (當前量 / 開盤至今分鐘) * 270
    若預估量 > 昨日量 * alert_threshold 則發出警示
    """
    minutes_elapsed = (now - market_open).total_seconds() / 60

    if minutes_elapsed < 1 or prev_close_volume <= 0:
        return {"minutes_elapsed": 0, "estimated_volume": 0, "vol_ratio": 0.0, "alert": False}

    estimated = int(current_volume / minutes_elapsed * 270)
    ratio = estimated / prev_close_volume

    return {
        "minutes_elapsed": round(minutes_elapsed, 1),
        "estimated_volume": estimated,
        "vol_ratio": round(ratio, 2),
        "alert": ratio > alert_threshold,
    }


# ═══════════════════════════════════════════════════════════
# 主掃描迴圈
# ═══════════════════════════════════════════════════════════

def scan_stocks(
    stock_ids: list,
    strategies: list,
    status: dict,
    fetch_price_fn,
    fetch_institutional_fn,
    get_name_fn,
    cancel_flag: dict | None = None,
    industry_map: dict | None = None,
) -> list:
    """
    主掃描迴圈。在背景執行緒中被呼叫。
    直接修改 status dict 以回報進度。
    cancel_flag: {"cancelled": bool} 用於外部取消。
    industry_map: dict[stock_id] → 產業名稱（如 "半導體業"）。
    """
    results = []
    total = len(stock_ids)

    # 透過 registry 判斷是否需要法人資料
    need_inst = any(
        get_strategy(code) and get_strategy(code).needs_institutional
        for code in strategies
    )
    ind_map = industry_map or {}

    for i, sid in enumerate(stock_ids, 1):
        # 檢查取消旗標（透過 status_ref 讀取 cancel 欄位）
        if cancel_flag:
            ref = cancel_flag.get("status_ref")
            if ref and ref.get("cancel"):
                status["current"] = "已取消"
                break

        status["current"] = f"策略掃描中：{sid} ({i}/{total})"
        status["progress"] = i - 1

        try:
            price_df = fetch_price_fn(sid)
            if price_df.empty or len(price_df) < 65:
                continue

            # 計算指標
            enriched = compute_screener_indicators(price_df)

            # 取得股票名稱 & 產業
            name = get_name_fn(sid, price_df)
            industry = ind_map.get(sid, "")

            # 基本資料
            last = enriched.iloc[-1]
            prev = enriched.iloc[-2] if len(enriched) >= 2 else last
            close = float(last["close"])
            change_pct = round((last["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0
            # 記錄資料日期，讓前端知道資料新鮮度
            data_date = ""
            if "date" in enriched.columns:
                data_date = str(enriched["date"].iloc[-1])[:10]
            elif hasattr(enriched.index, 'strftime'):
                data_date = enriched.index[-1].strftime("%Y-%m-%d")
            ma5  = round(float(last["s_ma5"]), 2) if pd.notna(last.get("s_ma5")) else None
            ma10 = round(float(last["s_ma10"]), 2) if pd.notna(last.get("s_ma10")) else None
            ma20 = round(float(last["s_ma20"]), 2) if pd.notna(last.get("s_ma20")) else None
            vol_ratio = round(float(last["s_vol_ratio"]), 2) if pd.notna(last.get("s_vol_ratio")) else None

            # 取得法人資料
            inst_df = pd.DataFrame()
            if need_inst:
                try:
                    inst_df = fetch_institutional_fn(sid)
                except Exception:
                    pass

            # ── 法人淨買賣超（供高檔出貨過濾，含前日資料） ──
            _foreign_net, _trust_net = None, None
            _prev_foreign_net, _prev_trust_net = None, None
            try:
                import finlab_fetcher as _flf
                _foreign_net, _trust_net, _prev_foreign_net, _prev_trust_net = _flf.get_institutional_net_2d(sid)
            except Exception:
                pass

            # ── 透過 registry 逐策略檢查 ──
            triggered = []
            details = {}

            for code in strategies:
                info = get_strategy(code)
                if info is None:
                    continue

                # 依據策略宣告的需求組裝參數
                strat_kwargs = {}
                if info.needs_industry:
                    strat_kwargs["industry"] = industry
                if info.needs_institutional:
                    strat_kwargs["inst_df"] = inst_df
                # 法人淨買賣超（高檔出貨過濾用）
                strat_kwargs["foreign_net"] = _foreign_net
                strat_kwargs["trust_net"] = _trust_net
                strat_kwargs["prev_foreign_net"] = _prev_foreign_net
                strat_kwargs["prev_trust_net"] = _prev_trust_net

                result = info.func(enriched, **strat_kwargs)
                if result:
                    triggered.append(code)
                    details[code] = result

            # ── 全策略通用：動能改良型基本面閘門 ──
            if triggered:
                profit_ok, profit_reason = check_profitability(sid)
                if not profit_ok:
                    strats = ",".join(triggered)
                    print(f"  ⛔ {sid} 技術面通過({strats})但基本面不合格，排除")
                    continue  # 整檔跳過，不加入結果

                labels = " / ".join(details[c]["label"] for c in triggered)
                # 轉機潛力股加上標籤
                if profit_reason == "轉機潛力股":
                    labels += " ⚡轉機潛力股"
                    print(f"  ⚡ {sid} {name} 轉機潛力股（虧損收斂 + 營收爆發）")

                # 策略 E 觸發時，加測朱家泓回檔買點（含四步診斷）
                pullback_buy = False
                pullback_steps = []
                if "E" in triggered:
                    from strategies.master_chu import check_pullback_buy_point
                    pb = check_pullback_buy_point(enriched)
                    pullback_steps = pb.get("steps", [])
                    if pb["triggered"]:
                        pullback_buy = True
                        labels += " 🎯回後買上漲點"
                        print(f"  🎯 {sid} {name} 朱家泓回後買上漲點 (黃金轉折)")

                # 決定追蹤池：Pool B 優先於 Pool A
                pool = None
                if "F" in triggered:
                    pool = "B"
                elif "E" in triggered:
                    pool = "A"

                results.append({
                    "stock_id": sid,
                    "name": name,
                    "close": close,
                    "change_pct": change_pct,
                    "data_date": data_date,
                    "ma5": ma5,
                    "ma10": ma10,
                    "ma20": ma20,
                    "vol_ratio": vol_ratio,
                    "industry": industry,
                    "triggered": triggered,
                    "strategy_labels": labels,
                    "strategy_details": details,
                    "profitability": profit_reason,
                    "pool": pool,
                    "pullback_buy": pullback_buy,
                    "pullback_steps": pullback_steps,
                    "exit_strategy": {
                        "short":  {"price": ma5,  "label": "跌破減碼"},
                        "medium": {"price": ma10, "label": "強勢支撐"},
                        "final":  {"price": ma20, "label": "全數清倉"},
                    },
                })

        except Exception as e:
            print(f"選股掃描 {sid} 失敗：{e}")
            continue

    status["progress"] = total
    return results
