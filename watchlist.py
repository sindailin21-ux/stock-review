"""
watchlist.py — AI 強勢成長股：雙池追蹤系統 V2.0
持久化 watchlist.json，追蹤池別轉換（A→B 轉強、移出等）。
"""
from __future__ import annotations

import json
import os
import pandas as pd

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), "watchlist.json")

_EMPTY_WATCHLIST = {
    "version": "2.0",
    "last_scan_date": "",
    "stocks": {},
}


# ── 讀寫 ──────────────────────────────────────────────────

def load_watchlist() -> dict:
    """讀取 watchlist.json，不存在回空結構。"""
    if not os.path.exists(WATCHLIST_FILE):
        return json.loads(json.dumps(_EMPTY_WATCHLIST))  # deep copy
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 版本相容
        if data.get("version") != "2.0":
            return json.loads(json.dumps(_EMPTY_WATCHLIST))
        return data
    except (json.JSONDecodeError, KeyError):
        return json.loads(json.dumps(_EMPTY_WATCHLIST))


def save_watchlist(data: dict) -> None:
    """儲存 watchlist 到 JSON。"""
    try:
        with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"💾 watchlist 已儲存：{WATCHLIST_FILE}")
    except Exception as e:
        print(f"⚠️ watchlist 儲存失敗：{e}")


# ── 移除條件 ──────────────────────────────────────────────

def check_removal(pool: str, close: float,
                  ma5: float | None, ma20: float | None) -> tuple[bool, str]:
    """
    檢查是否應從追蹤池移除。
    Pool A：收盤跌破 MA20 超過 3%。
    Pool B：收盤跌破 MA5，或偏離 MA20 超過 15%（過熱）。
    回傳 (should_remove, reason)。
    """
    if pool == "A":
        if ma20 and ma20 > 0 and close < ma20 * 0.97:
            return True, "破MA20逾3%"
    elif pool == "B":
        if ma5 and close < ma5:
            return True, "破MA5"
        if ma20 and ma20 > 0:
            deviation = (close - ma20) / ma20
            if deviation > 0.15:
                return True, "偏離MA20逾15%"
    return False, ""


# ── 核心：掃描後比對 ─────────────────────────────────────

def update_watchlist_after_scan(
    scan_rows: list[dict],
    scan_date: str,
    price_cache: dict | None = None,
    compute_fn=None,
) -> list[dict]:
    """
    比對前次 watchlist，為每筆 row 加上 pool_transition / removal_reason。
    同時將「本次未入選但仍在 watchlist」的股票做移除判斷。

    Parameters
    ----------
    scan_rows : 本次 scan_stocks() 回傳的結果列表
    scan_date : 本次掃描日期 "YYYY-MM-DD"
    price_cache : {stock_id: price_df} 用於計算移除條件
    compute_fn : compute_screener_indicators 函式（加指標用）

    Returns
    -------
    enriched_rows : 加上 pool_transition 欄位的 rows（含移出提醒列）
    """
    wl = load_watchlist()
    stocks = wl.get("stocks", {})

    # 建立本次入選清單
    current_map: dict[str, dict] = {}
    for row in scan_rows:
        sid = row.get("stock_id")
        if sid:
            current_map[sid] = row

    # ── 1. 為本次入選的 row 加上 pool_transition ──
    for row in scan_rows:
        sid = row["stock_id"]
        current_pool = row.get("pool")  # "A", "B", or None
        prev = stocks.get(sid)

        if prev and prev.get("status") == "active":
            prev_pool = prev.get("pool")
            if prev_pool == "A" and current_pool == "B":
                row["pool_transition"] = "轉強訊號"
            elif prev_pool == "B" and current_pool == "A":
                row["pool_transition"] = "降級觀察"
            else:
                row["pool_transition"] = None  # 穩定
        else:
            # 全新入選
            if current_pool == "A":
                row["pool_transition"] = "新進池A"
            elif current_pool == "B":
                row["pool_transition"] = "新進池B"
            else:
                row["pool_transition"] = None

        # 更新 watchlist entry
        if current_pool:
            from_pool = prev.get("pool") if prev else None
            transition_reason = (
                "momentum_confirmed" if (from_pool == "A" and current_pool == "B")
                else "downgrade" if (from_pool == "B" and current_pool == "A")
                else "stable" if prev and prev.get("status") == "active"
                else "initial_scan"
            )

            if sid not in stocks:
                # 新建 entry
                stocks[sid] = {
                    "pool": current_pool,
                    "entry_date": scan_date,
                    "last_scan_date": scan_date,
                    "status": "active",
                    "name": row.get("name", ""),
                    "industry": row.get("industry", ""),
                    "entry_close": row.get("close"),
                    "transitions": [
                        {"date": scan_date, "from": None, "to": current_pool,
                         "reason": "initial_scan"}
                    ],
                }
            else:
                entry = stocks[sid]
                old_pool = entry.get("pool")
                entry["last_scan_date"] = scan_date
                entry["status"] = "active"
                if old_pool != current_pool:
                    entry["pool"] = current_pool
                    entry["transitions"].append({
                        "date": scan_date,
                        "from": old_pool,
                        "to": current_pool,
                        "reason": transition_reason,
                    })

    # ── 2. 檢查前次 active 但本次未入選的股票 ──
    removed_rows = []
    for sid, entry in list(stocks.items()):
        if entry.get("status") != "active":
            continue
        if sid in current_map:
            continue  # 仍在名單中

        # 用 price_cache 計算移除條件
        close_val = None
        ma5_val = None
        ma20_val = None

        if price_cache and sid in price_cache:
            pdf = price_cache[sid]
            if not pdf.empty:
                if compute_fn:
                    try:
                        enriched = compute_fn(pdf.copy())
                        last = enriched.iloc[-1]
                        close_val = float(last["close"])
                        ma5_val = float(last.get("s_ma5", 0)) or None
                        ma20_val = float(last.get("s_ma20", 0)) or None
                    except Exception:
                        last = pdf.iloc[-1]
                        close_val = float(last["close"])
                else:
                    last = pdf.iloc[-1]
                    close_val = float(last["close"])

        pool = entry.get("pool", "A")

        if close_val is not None:
            should_remove, reason = check_removal(pool, close_val, ma5_val, ma20_val)
        else:
            # 無價格資料，標為暫離
            should_remove = True
            reason = "無最新價格資料"

        if should_remove:
            entry["status"] = "removed"
            entry["removal_date"] = scan_date
            entry["removal_reason"] = reason
            entry["transitions"].append({
                "date": scan_date,
                "from": pool,
                "to": None,
                "reason": f"removed_{reason}",
            })
            # 加一筆「移出」row 到結果中提醒用戶
            removed_rows.append({
                "stock_id": sid,
                "name": entry.get("name", ""),
                "close": close_val,
                "change_pct": 0,
                "ma5": ma5_val,
                "ma10": None,
                "ma20": ma20_val,
                "vol_ratio": None,
                "industry": entry.get("industry", ""),
                "triggered": [],
                "strategy_labels": f"⚠️ 移出（{reason}）",
                "strategy_details": {},
                "profitability": None,
                "exit_strategy": {},
                "pool": None,
                "pool_transition": "移出",
                "removal_reason": reason,
                "foreign_net": None,
                "trust_net": None,
                "rev_yoy": None,
            })
        else:
            # 未觸發移除 → 保留在 watchlist 但本次未入選（可能暫時不符合）
            # 不做任何狀態變更，等下次掃描再判斷
            pass

    # ── 3. 儲存 + 回傳 ──
    wl["last_scan_date"] = scan_date
    wl["stocks"] = stocks
    save_watchlist(wl)

    # 合併：本次入選 + 移出提醒
    all_rows = scan_rows + removed_rows
    return all_rows
