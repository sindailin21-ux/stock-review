"""
strategies/master_chu.py
策略 G：朱家泓老師進場法 + 每日持股覆盤出場檢查。

進場條件（選股掃描用）：
  1. 型態：頭頭高底底高
  2. 均線：四線多排 (MA5>MA10>MA20>MA60) + 斜率向上
  3. MA20 扣抵值低於現價（MA20 自然上升）
  4. 量確認：收盤站上 MA5，量 > VMA5 × 1.2

覆盤出場規則（每日持股檢查）：
  1. CLOSE < MA5 → 減碼
  2. 高位長黑K / 放量十字星 → 警覺
  3. MA5 斜率由升轉平 → 落袋
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from strategies import register, StrategyInfo
from strategies._helpers import detect_swing_highs_lows


# ═══════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════

def _compute_ma20_deduction(df: pd.DataFrame) -> dict:
    """
    計算 MA20 扣抵值分析。
    扣抵值 = 20 個交易日前的收盤價（明天會從 MA20 計算中「扣除」的值）。
    若扣抵值 < 現價 → MA20 自然上升（看漲）。
    """
    if len(df) < 21:
        return {"deduct_value": None, "is_bullish": False, "diff_pct": 0.0}

    current_close = float(df["close"].iloc[-1])
    deduct_value = float(df["close"].iloc[-20])
    diff_pct = (current_close - deduct_value) / deduct_value * 100 if deduct_value > 0 else 0.0

    return {
        "deduct_value": round(deduct_value, 2),
        "is_bullish": current_close > deduct_value,
        "diff_pct": round(diff_pct, 2),
    }


def _analyze_k_bar(df: pd.DataFrame) -> dict:
    """
    分析最新一根 K 棒的型態特徵。
    回傳：is_long_black, is_doji, body_pct, upper_shadow_pct, lower_shadow_pct, position_in_range
    """
    if len(df) < 2:
        return {
            "is_long_black": False, "is_doji": False,
            "body_pct": 0.0, "upper_shadow_pct": 0.0, "lower_shadow_pct": 0.0,
            "position_in_range": 50.0,
        }

    last = df.iloc[-1]
    o, h, l, c = float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"])

    # K 棒實體比例
    body = abs(c - o)
    body_pct = body / o * 100 if o > 0 else 0.0

    # 上下影線
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    upper_shadow_pct = upper_shadow / o * 100 if o > 0 else 0.0
    lower_shadow_pct = lower_shadow / o * 100 if o > 0 else 0.0

    # 位階：在近 60 日高低區間的百分位
    lookback = min(60, len(df))
    window = df.tail(lookback)
    high_60 = float(window["high"].max())
    low_60 = float(window["low"].min())
    if high_60 > low_60:
        position_in_range = (c - low_60) / (high_60 - low_60) * 100
    else:
        position_in_range = 50.0

    # 型態判定
    is_long_black = c < o and body_pct > 3.0   # 長黑K：收盤 < 開盤 + 實體 > 3%
    is_doji = body_pct < 0.5                     # 十字星：實體 < 0.5%

    return {
        "is_long_black": is_long_black,
        "is_doji": is_doji,
        "body_pct": round(body_pct, 2),
        "upper_shadow_pct": round(upper_shadow_pct, 2),
        "lower_shadow_pct": round(lower_shadow_pct, 2),
        "position_in_range": round(position_in_range, 1),
    }


def _compute_ma_slopes(df: pd.DataFrame, span: int = 3) -> dict:
    """
    計算各均線斜率（最近 span 日的差值）。
    """
    result = {}
    for p in [5, 10, 20, 60]:
        col = f"s_ma{p}"
        if col in df.columns and len(df) > span:
            now = df[col].iloc[-1]
            ago = df[col].iloc[-span]
            if pd.notna(now) and pd.notna(ago):
                result[f"ma{p}_slope"] = round(float(now - ago), 2)
            else:
                result[f"ma{p}_slope"] = 0.0
        else:
            result[f"ma{p}_slope"] = 0.0
    return result


# ═══════════════════════════════════════════════════════════
# 進場函式（選股掃描用）
# ═══════════════════════════════════════════════════════════

def strategy_g_chu_entry(df: pd.DataFrame, industry: str = "", **kwargs) -> dict | None:
    """
    朱家泓老師進場法：完整多頭結構確認。
    適用於全產業（不限 AI 產業）。

    條件：
    1. 型態：頭頭高底底高（近 40 日擺盪高低點遞升）
    2. 四線多排：MA5 > MA10 > MA20 > MA60，四條斜率皆 > 0
    3. MA20 扣抵值：扣抵值低於現價（MA20 自然看漲）
    4. 量確認：收盤站上 MA5，量 > VMA5 × 1.2
    """
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

    # 2. 四線多排 + 斜率向上
    if not (ma5 > ma10 > ma20 > ma60):
        return None

    slopes = _compute_ma_slopes(df, span=3)
    if not all(slopes[f"ma{p}_slope"] > 0 for p in [5, 10, 20, 60]):
        return None

    # 3. MA20 扣抵值
    deduction = _compute_ma20_deduction(df)
    if not deduction["is_bullish"]:
        return None

    # 4. 量確認
    if close <= ma5:
        return None
    if volume <= vol_ma5 * 1.2:
        return None

    return {
        "strategy": "G",
        "label": "朱家泓進場",
        "swing_pattern": True,
        "ma_alignment": True,
        "ma_slopes": slopes,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "ma20_deduct_value": deduction["deduct_value"],
        "ma20_deduct_bullish": deduction["is_bullish"],
        "ma20_deduct_diff_pct": deduction["diff_pct"],
        "vol_ratio": round(volume / vol_ma5, 2),
        "dist_ma20_pct": round((close - ma20) / ma20 * 100, 2),
    }


# ═══════════════════════════════════════════════════════════
# 回檔買點偵測（覆盤加值功能）
# ═══════════════════════════════════════════════════════════

def check_pullback_buy_point(df: pd.DataFrame) -> dict:
    """
    朱家泓「回後買上漲點」黃金轉折偵測。

    邏輯：股價回檔至 MA20 附近，縮量洗盤後止跌轉強，確認扣抵支撐。
    適用於「已經在上升趨勢中拉回」的標的，找進場 / 加碼時機。

    條件（全部滿足才觸發）：
    1. 回檔到位：收盤在 MA20 附近（MA20 ≤ Close ≤ MA20 × 1.05）
    2. 縮量洗盤：前三日中至少兩日量 < VMA5（主力不賣、浮額清洗）
    3. 轉強確認：Close > MA5 且 MA5 近 3 日斜率 ≥ 0（不再下彎）
    4. 扣抵支撐：Close > MA20 扣抵值（MA20 有上升慣性）

    回傳 {"triggered": bool, "detail": str, "steps": [...]}
    steps 永遠包含四步診斷，每步 {name, ok, value, desc}
    """
    # 預設四步骤空結果
    empty_steps = [
        {"name": "回檔到位", "ok": False, "value": "-", "desc": "資料不足"},
        {"name": "縮量洗盤", "ok": False, "value": "-", "desc": "資料不足"},
        {"name": "轉強確認", "ok": False, "value": "-", "desc": "資料不足"},
        {"name": "扣抵支撐", "ok": False, "value": "-", "desc": "資料不足"},
    ]
    result = {"triggered": False, "detail": "", "steps": empty_steps}

    if len(df) < 25:
        return result

    last = df.iloc[-1]
    close = float(last["close"])
    ma5 = last.get("s_ma5")
    ma20 = last.get("s_ma20")
    vol = float(last["volume"])
    vol_ma5 = last.get("s_vol_ma5")

    if any(pd.isna(v) for v in [ma5, ma20, vol_ma5]):
        return result
    ma5_f, ma20_f, vol_ma5_f = float(ma5), float(ma20), float(vol_ma5)

    if ma20_f == 0 or vol_ma5_f == 0:
        return result

    steps = []

    # ── 條件 1：回檔到位 ──
    dist_ma20 = round((close - ma20_f) / ma20_f * 100, 2)
    step1_ok = ma20_f <= close <= ma20_f * 1.05
    steps.append({
        "name": "回檔到位",
        "ok": step1_ok,
        "value": f"{dist_ma20:+.1f}%",
        "desc": f"收盤{close:.0f} vs MA20 {ma20_f:.0f}（需0~5%）",
    })

    # ── 條件 2：縮量洗盤（與前波爆量比） ──
    # 找近 20 根 K 線中的最大成交量作為「前波爆量」
    lookback_vol = min(20, len(df))
    peak_vol = float(df["volume"].iloc[-lookback_vol:].max())
    # 取回檔期間（前 3 日，不含今日）的平均量
    shrink_avg = 0.0
    if len(df) >= 4:
        shrink_vols = [float(df["volume"].iloc[i]) for i in range(-4, -1)]
        shrink_avg = sum(shrink_vols) / len(shrink_vols)
    # 計算縮量比例
    vol_ratio_pct = round(shrink_avg / peak_vol * 100, 1) if peak_vol > 0 else 100.0
    # 判定：需 ≤ 50%（初步洗盤門檻）
    step2_ok = vol_ratio_pct <= 50
    # 分級描述
    if vol_ratio_pct <= 25:
        shrink_level = "窒息量（轉折點）"
    elif vol_ratio_pct <= 40:
        shrink_level = "洗盤完成（準備發動）"
    elif vol_ratio_pct <= 50:
        shrink_level = "初步洗盤（觀察期）"
    else:
        shrink_level = "尚未縮量"
    steps.append({
        "name": "縮量洗盤",
        "ok": step2_ok,
        "value": f"{vol_ratio_pct:.0f}%",
        "desc": f"回檔均量/前波爆量={vol_ratio_pct:.0f}%（需≤50%）{shrink_level}",
    })

    # ── 條件 3：轉強確認 ──
    slopes = _compute_ma_slopes(df, span=3)
    ma5_slope = slopes.get("ma5_slope", 0.0)
    above_ma5 = close > ma5_f
    step3_ok = above_ma5 and ma5_slope >= 0
    if above_ma5:
        step3_desc = f"站上MA5，斜率{ma5_slope:+.1f}"
    else:
        step3_desc = f"收盤{close:.0f} < MA5 {ma5_f:.0f}"
    steps.append({
        "name": "轉強確認",
        "ok": step3_ok,
        "value": f"MA5斜率{ma5_slope:+.1f}",
        "desc": step3_desc,
    })

    # ── 條件 4：扣抵支撐 ──
    deduction = _compute_ma20_deduction(df)
    step4_ok = deduction["is_bullish"]
    deduct_val = deduction["deduct_value"]
    steps.append({
        "name": "扣抵支撐",
        "ok": step4_ok,
        "value": f"{deduct_val:.0f}" if deduct_val else "-",
        "desc": f"扣抵值{deduct_val:.0f} {'< 現價↑' if step4_ok else '> 現價↓'}" if deduct_val else "無資料",
    })

    result["steps"] = steps

    # ── 全部通過 → 觸發買點 ──
    if all(s["ok"] for s in steps):
        result["triggered"] = True
        result["detail"] = (
            f"回檔至MA20附近(距{dist_ma20:+.1f}%)，"
            f"縮量比{vol_ratio_pct:.0f}%({shrink_level})，"
            f"站上MA5(斜率{ma5_slope:+.1f})，"
            f"扣抵值{deduct_val:.0f}(↑)"
        )
    return result


# ═══════════════════════════════════════════════════════════
# 覆盤函式（每日持股出場檢查）
# ═══════════════════════════════════════════════════════════

def chu_daily_review(df: pd.DataFrame, **kwargs) -> dict:
    """
    朱家泓老師每日持股覆盤：檢查出場/減碼/警覺訊號。
    永遠回傳結果（不回 None），因為每檔持股都需要給出判斷。

    規則（依嚴重度排序）：
    1. CLOSE < MA5 → status=reduce, 指令：減碼
    2. 高位長黑K（body>3%, close<open, 位階>80%）→ status=alert, 指令：警覺
    3. 高位放量十字星（body<0.5%, vol>1.5x, 位階>80%）→ status=alert, 指令：警覺
    4. MA5 斜率由升轉平 → status=take_profit, 指令：落袋
    5. 以上皆無 → status=healthy, 指令：續抱
    """
    signals = []
    status = "healthy"

    if len(df) < 10:
        return {
            "status": "healthy",
            "signals": [],
            "ma_status": {},
            "k_bar": {},
            "summary": "資料不足，無法判斷",
        }

    last = df.iloc[-1]
    close = float(last["close"])
    ma5 = last.get("s_ma5")
    vol_ratio_val = last.get("s_vol_ratio")

    # ── MA 狀態 ──
    slopes = _compute_ma_slopes(df, span=3)
    deduction = _compute_ma20_deduction(df)

    ma5_f = float(ma5) if pd.notna(ma5) else None
    ma10_f = float(last.get("s_ma10")) if pd.notna(last.get("s_ma10")) else None
    ma20_f = float(last.get("s_ma20")) if pd.notna(last.get("s_ma20")) else None
    ma60_f = float(last.get("s_ma60")) if pd.notna(last.get("s_ma60")) else None

    # 判定均線排列
    if all(v is not None for v in [ma5_f, ma10_f, ma20_f, ma60_f]):
        if ma5_f > ma10_f > ma20_f > ma60_f:
            alignment = "四線多排"
        elif ma5_f < ma10_f < ma20_f < ma60_f:
            alignment = "四線空排"
        else:
            alignment = "均線糾結"
    else:
        alignment = "資料不足"

    ma_status = {
        "alignment": alignment,
        "ma5": ma5_f,
        "ma10": ma10_f,
        "ma20": ma20_f,
        "ma60": ma60_f,
        "ma5_slope": slopes.get("ma5_slope", 0.0),
        "ma20_slope": slopes.get("ma20_slope", 0.0),
        "ma20_deduct_value": deduction.get("deduct_value"),
        "ma20_deduct_bullish": deduction.get("is_bullish", False),
        "ma20_deduct_diff_pct": deduction.get("diff_pct", 0.0),
    }

    # ── K 棒分析 ──
    k_bar = _analyze_k_bar(df)

    vol_ratio = float(vol_ratio_val) if pd.notna(vol_ratio_val) else 0.0

    # ── 規則 1：CLOSE < MA5 → 減碼 ──
    if ma5_f is not None and close < ma5_f:
        signals.append({
            "type": "reduce",
            "rule": "收盤跌破5日線",
            "detail": f"收盤 {close:.2f} < MA5 {ma5_f:.2f}",
        })
        status = "reduce"

    # ── 規則 2：高位長黑K → 警覺 ──
    if k_bar["is_long_black"] and k_bar["position_in_range"] > 80:
        signals.append({
            "type": "alert",
            "rule": "高位長黑K",
            "detail": f"實體 {k_bar['body_pct']:.1f}%，位階 {k_bar['position_in_range']:.0f}%",
        })
        if status == "healthy":
            status = "alert"

    # ── 規則 3：高位放量十字星 → 警覺 ──
    if k_bar["is_doji"] and vol_ratio > 1.5 and k_bar["position_in_range"] > 80:
        signals.append({
            "type": "alert",
            "rule": "高位量大十字星",
            "detail": f"量比 {vol_ratio:.1f}x，位階 {k_bar['position_in_range']:.0f}%",
        })
        if status == "healthy":
            status = "alert"

    # ── 規則 4：MA5 斜率由升轉平 → 落袋 ──
    if len(df) > 5:
        # 前一段斜率（-5 到 -3）
        ma5_col = df["s_ma5"]
        if len(ma5_col) > 5 and pd.notna(ma5_col.iloc[-5]) and pd.notna(ma5_col.iloc[-3]):
            prev_slope = float(ma5_col.iloc[-3] - ma5_col.iloc[-5])
            curr_slope = slopes.get("ma5_slope", 0.0)

            # 前段上升，現段趨平或下彎
            if prev_slope > 0 and curr_slope <= 0.05 * float(ma5_f if ma5_f else 1) / 100:
                signals.append({
                    "type": "take_profit",
                    "rule": "5日線走平",
                    "detail": f"前段斜率 {prev_slope:+.2f} → 現段 {curr_slope:+.2f}",
                })
                if status == "healthy":
                    status = "take_profit"

    # ── 規則 5：回檔買點偵測 ──
    pullback = check_pullback_buy_point(df)
    pullback_triggered = pullback["triggered"]
    if pullback_triggered:
        signals.insert(0, {
            "type": "buy_point",
            "rule": "🎯 回後買上漲點 (黃金轉折)",
            "detail": pullback["detail"],
        })

    # ── 摘要 ──
    if pullback_triggered:
        summary = "🎯 朱家泓：回後買上漲點 (黃金轉折)"
        if len(signals) > 1:
            other = "；".join(s["rule"] for s in signals[1:])
            summary += f"（同時：{other}）"
        status = "buy_point"
    elif not signals:
        summary = "持股健康，續抱"
    else:
        summary = "；".join(s["rule"] for s in signals)
        status_label = {"reduce": "建議減碼", "alert": "提高警覺", "take_profit": "建議落袋"}
        summary = f"{status_label.get(status, '')}：{summary}"

    return {
        "status": status,
        "signals": signals,
        "ma_status": ma_status,
        "k_bar": k_bar,
        "summary": summary,
        "pullback_buy": pullback_triggered,
    }


# ═══════════════════════════════════════════════════════════
# 策略 H：朱家泓最佳版（ADX 趨勢 + RSI 濾鏡）
# ═══════════════════════════════════════════════════════════

def strategy_h_chu_best(df: pd.DataFrame, industry: str = "", **kwargs) -> dict | None:
    """
    朱家泓進場法 — 最佳版（回測年化 +37%）。

    與原版 G 的差異：
      1. 趨勢偵測：ADX(8)>20 且 +DI>-DI（取代擺盪高低點）
      2. RSI(14) 濾鏡：50 < RSI < 80（趨勢中但未超買）
      3. 放寬量能：量 > VMA5 × 1.2（不要求紅K/上影線）
      4. 動態乖離：ADX≤30→5%, ADX≤40→8%, ADX>40→10%

    出場建議：移動停利 15%（從最高點回落），停損 6%，最低持有 5 天。
    """
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

    # ADX / RSI 指標
    adx = last.get("s_adx14")
    plus_di = last.get("s_plus_di14")
    minus_di = last.get("s_minus_di14")
    rsi14 = last.get("s_rsi14")

    if any(pd.isna(v) for v in [ma5, ma10, ma20, ma60, vol_ma5]):
        return None
    ma5, ma10, ma20, ma60, vol_ma5 = (
        float(ma5), float(ma10), float(ma20), float(ma60), float(vol_ma5),
    )
    if ma60 == 0 or vol_ma5 == 0:
        return None

    # 1. ADX(8) 趨勢確認：ADX>20 且上升趨勢（+DI > -DI）
    if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
        return None
    adx, plus_di, minus_di = float(adx), float(plus_di), float(minus_di)
    if adx <= 20:
        return None
    if plus_di <= minus_di:
        return None

    # 2. RSI(14) 濾鏡：50-80（趨勢進行中但未超買）
    if pd.isna(rsi14):
        return None
    rsi14 = float(rsi14)
    if rsi14 <= 50 or rsi14 >= 80:
        return None

    # 3. 四線多排 + 斜率向上
    if not (ma5 > ma10 > ma20 > ma60):
        return None

    slopes = _compute_ma_slopes(df, span=3)
    if not all(slopes[f"ma{p}_slope"] > 0 for p in [5, 10, 20, 60]):
        return None

    # 4. MA20 扣抵值
    deduction = _compute_ma20_deduction(df)
    if not deduction["is_bullish"]:
        return None

    # 5. 量確認（放寬版：量 > VMA5 × 1.2，不要求紅K/上影線）
    if close <= ma5:
        return None
    if volume <= vol_ma5 * 1.2:
        return None

    # 6. 動態乖離率上限（依 ADX 趨勢強度放寬）
    #    ADX 20-30（溫和趨勢）：≤ 5%
    #    ADX 30-40（強趨勢）  ：≤ 8%
    #    ADX > 40 （極強趨勢）：≤ 10%
    bias = (close - ma5) / ma5
    if adx <= 30:
        max_bias = 0.05
    elif adx <= 40:
        max_bias = 0.08
    else:
        max_bias = 0.10
    if bias >= max_bias:
        return None

    # ── 計算進場 / 出場價位 ──
    entry_price = close                             # 進場價 = 今日收盤
    stop_loss_price = round(close * 0.94, 2)        # 停損價 = -6%
    trail_start_price = round(close * 1.15, 2)      # 停利起算 = +15%（之後從最高點回落15%出場）
    risk_per_share = round(close - stop_loss_price, 2)  # 每股風險
    reward_target = round(trail_start_price - close, 2) # 每股目標獲利
    rr_ratio = round(reward_target / risk_per_share, 1) if risk_per_share > 0 else 0  # 風報比

    return {
        "strategy": "H",
        "label": "朱家泓最佳",
        "adx": round(adx, 1),
        "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1),
        "rsi14": round(rsi14, 1),
        "trend_strength": "強" if adx > 30 else "中",
        "ma_alignment": True,
        "ma_slopes": slopes,
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "ma20_deduct_value": deduction["deduct_value"],
        "ma20_deduct_bullish": deduction["is_bullish"],
        "ma20_deduct_diff_pct": deduction["diff_pct"],
        "vol_ratio": round(volume / vol_ma5, 2),
        "dist_ma20_pct": round((close - ma20) / ma20 * 100, 2),
        "bias_ma5_pct": round(bias * 100, 2),
        "bias_max_pct": round(max_bias * 100, 1),
        # ── 進出場計畫 ──
        "entry_price": round(entry_price, 2),
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct": 6,
        "trail_stop_pct": 15,
        "trail_start_price": trail_start_price,
        "min_hold_days": 5,
        "risk_per_share": risk_per_share,
        "rr_ratio": rr_ratio,
    }


# ═══════════════════════════════════════════════════════════
# H 策略逐條診斷
# ═══════════════════════════════════════════════════════════

def diagnose_h_strategy(df: pd.DataFrame, industry: str = "", **kwargs) -> dict:
    """
    H 策略逐條診斷 — 不管通過與否都回傳完整結果。
    與 strategy_h_chu_best() 邏輯完全一致，但不提前 return None。
    """
    checks = []
    summary = {}

    if len(df) < 65:
        return {"passed": False, "checks": [], "summary": {},
                "error": "歷史資料不足（需至少 65 天）"}

    last = df.iloc[-1]
    close = float(last["close"])
    ma5 = last.get("s_ma5")
    ma10 = last.get("s_ma10")
    ma20 = last.get("s_ma20")
    ma60 = last.get("s_ma60")
    volume = float(last["volume"])
    vol_ma5 = last.get("s_vol_ma5")

    adx = last.get("s_adx14")
    plus_di = last.get("s_plus_di14")
    minus_di = last.get("s_minus_di14")
    rsi14 = last.get("s_rsi14")

    # 安全轉換
    _has_ma = not any(pd.isna(v) for v in [ma5, ma10, ma20, ma60, vol_ma5])
    if _has_ma:
        ma5, ma10, ma20, ma60, vol_ma5 = (
            float(ma5), float(ma10), float(ma20), float(ma60), float(vol_ma5),
        )
    else:
        ma5 = ma10 = ma20 = ma60 = vol_ma5 = 0.0

    _has_adx = not any(pd.isna(v) for v in [adx, plus_di, minus_di]) if adx is not None else False
    if _has_adx:
        adx, plus_di, minus_di = float(adx), float(plus_di), float(minus_di)
    else:
        adx = plus_di = minus_di = 0.0

    _has_rsi = rsi14 is not None and not pd.isna(rsi14)
    rsi14 = float(rsi14) if _has_rsi else 0.0

    summary = {
        "close": close, "ma5": round(ma5, 2), "ma10": round(ma10, 2),
        "ma20": round(ma20, 2), "ma60": round(ma60, 2),
        "volume": int(volume), "vol_ma5": int(vol_ma5),
        "adx": round(adx, 1), "plus_di": round(plus_di, 1),
        "minus_di": round(minus_di, 1), "rsi14": round(rsi14, 1),
    }

    # ① ADX(8) > 20
    adx_ok = adx > 20
    checks.append({
        "name": "ADX(8) > 20",
        "passed": adx_ok,
        "value": f"{adx:.1f}",
        "threshold": "> 20",
        "detail": "強趨勢" if adx > 30 else ("中趨勢" if adx_ok else "無趨勢/盤整"),
    })

    # ② +DI > -DI
    di_ok = plus_di > minus_di
    checks.append({
        "name": "+DI > -DI",
        "passed": di_ok,
        "value": f"+DI={plus_di:.1f}, -DI={minus_di:.1f}",
        "threshold": "+DI > -DI",
        "detail": "上升趨勢" if di_ok else "下降趨勢",
    })

    # ③ RSI(14) 50-80
    rsi_ok = 50 < rsi14 < 80
    checks.append({
        "name": "RSI(14) 50-80",
        "passed": rsi_ok,
        "value": f"{rsi14:.1f}",
        "threshold": "50 < RSI < 80",
        "detail": "超買" if rsi14 >= 80 else ("動能不足" if rsi14 <= 50 else "趨勢中"),
    })

    # ④ 四線多排
    ma_align = ma5 > ma10 > ma20 > ma60 if _has_ma else False
    checks.append({
        "name": "四線多排",
        "passed": ma_align,
        "value": f"MA5={ma5:.2f} > MA10={ma10:.2f} > MA20={ma20:.2f} > MA60={ma60:.2f}",
        "threshold": "MA5 > MA10 > MA20 > MA60",
        "detail": "均線排列正確" if ma_align else "均線未多排",
    })

    # ⑤ 四線斜率全 > 0
    slopes = _compute_ma_slopes(df, span=3) if _has_ma else {"ma5_slope": 0, "ma10_slope": 0, "ma20_slope": 0, "ma60_slope": 0}
    slopes_ok = all(slopes[f"ma{p}_slope"] > 0 for p in [5, 10, 20, 60])
    failed_slopes = [f"MA{p}" for p in [5, 10, 20, 60] if slopes[f"ma{p}_slope"] <= 0]
    checks.append({
        "name": "四線斜率全 > 0",
        "passed": slopes_ok,
        "value": ", ".join(f"MA{p}={slopes[f'ma{p}_slope']:.2f}" for p in [5, 10, 20, 60]),
        "threshold": "全部 > 0",
        "detail": "全部向上" if slopes_ok else f"{','.join(failed_slopes)} 向下",
    })
    summary["slopes"] = {f"ma{p}": round(slopes[f"ma{p}_slope"], 4) for p in [5, 10, 20, 60]}

    # ⑥ MA20 扣抵看多
    deduction = _compute_ma20_deduction(df) if _has_ma else {"is_bullish": False, "deduct_value": 0, "diff_pct": 0}
    checks.append({
        "name": "MA20 扣抵看多",
        "passed": deduction["is_bullish"],
        "value": f"扣抵值={deduction['deduct_value']}, 差異={deduction['diff_pct']}%",
        "threshold": "扣抵值 < 現價",
        "detail": "MA20 將上升" if deduction["is_bullish"] else "MA20 有下彎壓力",
    })

    # ⑦ 收盤 > MA5
    above_ma5 = close > ma5 if _has_ma else False
    checks.append({
        "name": "收盤 > MA5",
        "passed": above_ma5,
        "value": f"{close:.2f} vs MA5={ma5:.2f}",
        "threshold": f"> {ma5:.2f}",
        "detail": "站穩均線上" if above_ma5 else "跌破5日線",
    })

    # ⑧ 量 > VMA5 × 1.2
    vol_threshold = vol_ma5 * 1.2 if vol_ma5 > 0 else 0
    vol_ok = volume > vol_threshold if vol_ma5 > 0 else False
    vol_pct = round(volume / vol_threshold * 100, 1) if vol_threshold > 0 else 0
    checks.append({
        "name": "量 > VMA5 x 1.2",
        "passed": vol_ok,
        "value": f"{int(volume):,} vs 需要 {int(vol_threshold):,}",
        "threshold": f"> {int(vol_threshold):,}",
        "detail": f"達標 {vol_pct}%" if vol_ok else f"僅 {vol_pct}%",
    })

    # ⑨ 動態乖離率（依 ADX 趨勢強度）
    bias = (close - ma5) / ma5 if ma5 > 0 else 0
    if adx <= 30:
        diag_max_bias = 0.05
    elif adx <= 40:
        diag_max_bias = 0.08
    else:
        diag_max_bias = 0.10
    bias_ok = bias < diag_max_bias
    bias_label = f"乖離率 < {diag_max_bias*100:.0f}%"
    checks.append({
        "name": bias_label,
        "passed": bias_ok,
        "value": f"{bias*100:.2f}%",
        "threshold": f"< {diag_max_bias*100:.0f}%（ADX={adx:.0f}）",
        "detail": "合理範圍" if bias_ok else "乖離過大，追高風險",
    })

    all_passed = all(c["passed"] for c in checks)
    passed_count = sum(1 for c in checks if c["passed"])

    return {
        "passed": all_passed,
        "passed_count": passed_count,
        "total_checks": len(checks),
        "checks": checks,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════
# 策略註冊
# ═══════════════════════════════════════════════════════════

register(StrategyInfo(
    code="G",
    name="朱家泓進場",
    description="頭頭高底底高 + 四線多排 + MA20扣抵值",
    category="master_chu",
    pool=None,
    needs_institutional=False,
    needs_industry=True,
    css_class="strat-g",
    func=strategy_g_chu_entry,
    has_review_mode=True,
    review_func=chu_daily_review,
))

register(StrategyInfo(
    code="H",
    name="朱家泓最佳",
    description="ADX(8)趨勢 + RSI濾鏡 + 四線多排（回測年化+37%）",
    category="master_chu",
    pool=None,
    needs_institutional=False,
    needs_industry=True,
    css_class="strat-h",
    func=strategy_h_chu_best,
    has_review_mode=True,
    review_func=chu_daily_review,
))
