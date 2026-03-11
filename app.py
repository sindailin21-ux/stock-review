"""
app.py
台股覆盤工具 v1 - 整合版 Flask 應用

路由架構：
  /                    單支股票查詢頁
  /portfolio           持股清單管理頁
  /batch               批次分析頁
  /screener            選股頁

  GET  /api/portfolio          取得持股清單
  POST /api/portfolio          新增/更新單筆
  DEL  /api/portfolio/<id>     刪除單筆
  POST /api/portfolio/import   CSV 匯入
  GET  /api/portfolio/export   CSV 匯出下載

  POST /api/query              單支股票 AI 分析
  GET  /api/history            查詢歷史

  POST /api/batch/run          觸發批次分析（背景執行）
  GET  /api/batch/status       輪詢批次進度

  POST /api/screener/run            觸發策略選股掃描（背景執行）
  GET  /api/screener/status         輪詢選股進度
  POST /api/screener/monitor/run    手動觸發盤中量能檢查
  GET  /api/screener/monitor/status 盤中量能檢查結果

  POST /api/telegram/single    傳送單支分析到 Telegram
  POST /api/telegram/batch     傳送批次摘要到 Telegram

啟動：
  本機：python3 app.py
  正式：gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
"""

import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, Response

from config import FINMIND_TOKEN, ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
import portfolio_store as ps

app = Flask(__name__)


@app.errorhandler(500)
def handle_500(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"伺服器錯誤：{e}"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": f"未預期錯誤：{e}"}), 500


def _sanitize_for_json(obj):
    """遞迴將 numpy 型別轉為 Python 原生型別，確保 JSON 可序列化"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# ═══════════════════════════════════════════════════════════
# 快取設定
# ═══════════════════════════════════════════════════════════

BATCH_CACHE_DIR = Path("cache/batch")   # 批次 AI 結果（磁碟，跨重啟保留）
QUERY_CACHE_DIR = Path("cache/query")   # 單支查詢（磁碟，跨重啟保留）

_query_cache: dict = {}   # 記憶體索引，key: "{stock_id}_{date}"


# ── 批次快取：磁碟讀寫 ────────────────────────────────────

def _batch_cache_path(today: str) -> Path:
    BATCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return BATCH_CACHE_DIR / f"ai_{today}.json"

def _load_batch_cache(today: str) -> dict:
    path = _batch_cache_path(today)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"📦 讀取批次 AI 快取：{path}（共 {len(data)} 檔）")
            return data
        except Exception:
            pass
    return {}

def _save_batch_cache(today: str, cache: dict):
    path = _batch_cache_path(today)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"💾 批次 AI 快取已儲存：{path}")

_batch_cache_date = ""   # 不再需要，保留供 _run_batch global 宣告相容

_screener_status = {
    "running":  False,
    "progress": 0,
    "total":    0,
    "current":  "",
    "done":     False,
    "error":    None,
    "rows":     [],
    "strategies": [],
    "cancel":   False,
}

_monitor_status = {
    "running":  False,
    "done":     False,
    "error":    None,
    "last_check": None,
    "alerts":   [],
    "stock_ids": [],
}

_intraday_status = {
    "running":  False,
    "done":     False,
    "progress": 0,
    "total":    0,
    "current":  "",
    "error":    None,
    "rows":     [],
}

_batch_status = {
    "running":   False,
    "progress":  0,
    "total":     0,
    "current":   "",
    "done":      False,
    "error":     None,
    "result":    [],      # list of stock_data dicts
    "report_html": "",
}


# ═══════════════════════════════════════════════════════════
# 工具函式
# ═══════════════════════════════════════════════════════════

def _build_chart_data(price_df):
    tail  = price_df.tail(30).copy()
    dates = tail["date"].dt.strftime("%m/%d").tolist()
    ohlcv = [
        {"o": float(r.open), "h": float(r.high), "l": float(r.low),
         "c": float(r.close), "v": int(r.volume)}
        for r in tail.itertuples()
    ]
    def _safe(col):
        s = tail.get(col, pd.Series(dtype=float))
        return [round(float(v), 4) if v == v else None for v in s]
    return {
        "dates":       dates,
        "ohlcv":       ohlcv,
        "macd_line":   _safe("macd"),
        "macd_signal": _safe("macd_signal"),
        "macd_hist":   _safe("macd_hist"),
        "rsi":         _safe("rsi"),
        "ma5":         _safe("ma5"),
        "ma10":        _safe("ma10"),
        "ma20":        _safe("ma20"),
    }


def _get_stock_name(stock_id: str, price_df) -> str:
    """從 FinMind API 或 price_df 取得股票名稱"""
    if "stock_name" in price_df.columns:
        return price_df["stock_name"].iloc[-1]
    if "name" in price_df.columns:
        return price_df["name"].iloc[-1]
    try:
        import requests as req
        resp = req.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == 200:
            for item in data.get("data", []):
                if item.get("stock_id") == stock_id:
                    return item.get("stock_name", stock_id)
    except Exception:
        pass
    return stock_id


# ═══════════════════════════════════════════════════════════
# 單支股票查詢邏輯
# ═══════════════════════════════════════════════════════════

def _query_stock(stock_id: str):
    """執行單支股票分析，回傳 (stock_data, error_msg)"""
    from data_fetcher import fetch_all
    from indicators import add_all_indicators, get_latest_signals
    from fundamentals import get_full_fundamental_summary
    from ai_analyzer import analyze_stock
    from report_generator import generate_stock_card

    stock_id = str(stock_id).zfill(4)

    raw      = fetch_all(stock_id)
    price_df = raw.get("price", pd.DataFrame())
    if price_df.empty:
        return None, f"找不到 {stock_id} 的資料，請確認股票代號是否正確"

    price_df     = add_all_indicators(price_df)
    signals      = get_latest_signals(price_df)
    fund_summary = get_full_fundamental_summary(raw)
    chart_data   = _build_chart_data(price_df)
    name         = _get_stock_name(stock_id, price_df)

    # 若持股清單有此股票，帶入成本資訊
    holding = ps.get_by_id(stock_id)
    portfolio_info = {
        "cost_price": holding["cost_price"] if holding else 0,
        "shares":     holding["shares"]     if holding else 0,
    }

    analysis = analyze_stock(stock_id, name, portfolio_info, signals, fund_summary)

    stock_data = {
        "stock_id":   stock_id,
        "name":       name,
        "portfolio":  portfolio_info,
        "signals":    signals,
        "analysis":   analysis,
        "chart_data": chart_data,
        "fundamentals": fund_summary,
    }
    card_html = generate_stock_card(stock_data)
    return {**stock_data, "card_html": card_html}, None


# ═══════════════════════════════════════════════════════════
# 批次分析邏輯（背景執行緒）
# ═══════════════════════════════════════════════════════════

def _run_batch(portfolio_df: pd.DataFrame, force: bool, today: str):
    global _batch_status, _batch_cache_date

    from data_fetcher import fetch_all
    from indicators import add_all_indicators, get_latest_signals
    from fundamentals import get_full_fundamental_summary
    from ai_analyzer import analyze_stock
    from report_generator import generate_report

    # 從磁碟讀取今日 AI 快取
    ai_cache = {} if force else _load_batch_cache(today)
    ai_cache_updated = dict(ai_cache)

    results = []
    total   = len(portfolio_df)

    for i, row in enumerate(portfolio_df.itertuples(), 1):
        stock_id = str(row.stock_id)
        name     = str(row.name)
        shares   = float(row.shares)
        cost     = float(row.cost_price)

        _batch_status["current"]  = f"{stock_id} {name}"
        _batch_status["progress"] = i - 1

        try:
            raw      = fetch_all(stock_id)
            price_df = raw.get("price", pd.DataFrame())
            if price_df.empty:
                continue

            price_df     = add_all_indicators(price_df)
            signals      = get_latest_signals(price_df)
            fund_summary = get_full_fundamental_summary(raw)
            chart_data   = _build_chart_data(price_df)

            portfolio_info = {"cost_price": cost, "shares": shares}

            if stock_id in ai_cache and not force:
                print(f"  ✅ {stock_id} 使用快取 AI 結果")
                analysis = ai_cache[stock_id]
            else:
                print(f"  🤖 {stock_id} AI 分析中...")
                analysis = analyze_stock(stock_id, name, portfolio_info, signals, fund_summary)
                ai_cache_updated[stock_id] = analysis

            results.append({
                "stock_id":   stock_id,
                "name":       name,
                "portfolio":  portfolio_info,
                "signals":    signals,
                "analysis":   analysis,
                "chart_data": chart_data,
                "fundamentals": fund_summary,
            })
        except Exception as e:
            print(f"批次分析 {stock_id} 失敗：{e}")
            continue

    # 儲存更新的 AI 快取到磁碟
    if ai_cache_updated != ai_cache or force:
        _save_batch_cache(today, ai_cache_updated)

    _batch_status["progress"]    = total
    _batch_status["result"]      = results
    _batch_status["report_html"] = generate_report(results, datetime.today().strftime("%Y/%m/%d")) if results else ""
    _batch_status["done"]        = True
    _batch_status["running"]     = False
    _batch_status["current"]     = ""


# ═══════════════════════════════════════════════════════════
# 選股邏輯（背景執行緒）
# ═══════════════════════════════════════════════════════════

def _run_screener(strategies: list, target_date: str = None, force_refresh: bool = False):
    """
    全自動選股流程（100% TWSE + TPEX 公開資料，不使用 FinMind）：
    Phase 1: TWSE/TPEX 全市場日行情 → 流動性過濾
    Phase 2: 逐股載入歷史行情（TWSE STOCK_DAY / TPEX 個股月報，約 5 個月）
    Phase 3: 批次載入法人籌碼（若策略 D 啟用）
    Phase 4: 逐股跑策略掃描（全部從 cache 讀取）

    force_refresh: 若為 True，忽略本地快取，強制重新抓取資料。
    """
    global _screener_status

    import pickle, os, hashlib

    from data_fetcher import fetch_market_daily
    from data_fetcher import fetch_stock_prices_batch, fetch_institutional_batch, fetch_industry_map, fetch_institutional_single
    from data_fetcher import fetch_revenue
    from fundamentals import get_revenue_summary
    from screener import scan_stocks

    # ── 本地快取（同日不重抓 TWSE/TPEX）──
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # ── Phase 1: 全市場行情 + 流動性過濾 ──
        _screener_status["current"] = "載入全市場行情（TWSE + TPEX）..."
        market_df = fetch_market_daily(target_date)
        if _screener_status["cancel"]:
            _screener_status["error"] = "已取消"
            return
        if market_df.empty:
            _screener_status["error"] = "無法載入市場日行情（可能為非交易日）"
            return

        # 確保數值欄位
        for col in ["Trading_Volume", "Trading_Money"]:
            if col in market_df.columns:
                market_df[col] = pd.to_numeric(market_df[col], errors="coerce").fillna(0)

        # 只保留一般股票（4位數字代號 1xxx-9xxx），排除 ETF(00xx)、權證等
        stock_mask = market_df["stock_id"].str.match(r"^[1-9]\d{3}$")
        market_df = market_df[stock_mask].copy()
        total_stocks = len(market_df)
        print(f"📋 一般股票共 {total_stocks} 檔")

        _screener_status["current"] = "流動性過濾（成交額/量）..."

        # 過濾：成交金額 > 1 億 OR 成交量 > 1,000,000 股（= 1000 張）
        vol_col = "Trading_Volume" if "Trading_Volume" in market_df.columns else None
        money_col = "Trading_Money" if "Trading_Money" in market_df.columns else None

        if vol_col and money_col:
            mask = (market_df[money_col] > 100_000_000) | (market_df[vol_col] > 1_000_000)
        elif vol_col:
            mask = market_df[vol_col] > 1_000_000
        elif money_col:
            mask = market_df[money_col] > 100_000_000
        else:
            _screener_status["error"] = "行情資料缺少成交量/金額欄位"
            return

        final_ids = sorted(market_df[mask]["stock_id"].tolist())
        print(f"💧 流動性過濾後 {len(final_ids)} 檔（原 {total_stocks} 檔）")

        if not final_ids:
            _screener_status["error"] = "過濾後無符合條件的股票"
            return

        # ── 快取 key：用 target_date 或今日日期 ──
        from datetime import datetime as _dt
        cache_date = target_date or _dt.today().strftime("%Y-%m-%d")
        cache_key  = f"scan_{cache_date}"
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

        # 建立名稱對照表 + 交易所對照表
        name_map = dict(zip(market_df["stock_id"], market_df["name"])) if "name" in market_df.columns else {}
        exchange_map = dict(zip(market_df["stock_id"], market_df["exchange"])) if "exchange" in market_df.columns else {}

        # ── Phase 1.5: 產業分類 ──
        _screener_status["current"] = "載入產業分類..."
        industry_map = fetch_industry_map()

        # ── Phase 2 & 3: 歷史行情 + 法人籌碼（有快取直接載入）──
        use_cache = os.path.exists(cache_file) and not force_refresh

        if force_refresh and os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                print(f"🔄 強制重抓：已刪除快取 {cache_file}")
            except Exception as e:
                print(f"⚠️ 無法刪除快取：{e}")

        if use_cache:
            # 讀取快取建立時間
            cache_mtime = os.path.getmtime(cache_file)
            cache_ts = _dt.fromtimestamp(cache_mtime).strftime("%Y-%m-%d %H:%M:%S")
            _screener_status["cache_info"] = {
                "from_cache": True,
                "cache_time": cache_ts,
                "cache_date": cache_date,
            }
            _screener_status["current"] = f"⚡ 載入本地快取（{cache_date}，建立於 {cache_ts}）..."
            print(f"⚡ 使用快取：{cache_file}（建立於 {cache_ts}）")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            price_cache = cached["price_cache"]
            inst_cache  = cached["inst_cache"]
        else:
            # Phase 2: 逐股載入歷史行情
            if _screener_status["cancel"]:
                _screener_status["error"] = "已取消"
                return
            price_cache = fetch_stock_prices_batch(
                stock_ids=final_ids,
                exchange_map=exchange_map,
                target_date=target_date,
                months=5,
                status=_screener_status,
            )

            # Phase 3: 批次載入法人籌碼
            inst_cache = {}
            if _screener_status["cancel"]:
                _screener_status["error"] = "已取消"
                return
            inst_cache = fetch_institutional_batch(target_date, 5, _screener_status)

            # 寫入快取
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"price_cache": price_cache, "inst_cache": inst_cache}, f)
                cache_ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
                _screener_status["cache_info"] = {
                    "from_cache": False,
                    "cache_time": cache_ts,
                    "cache_date": cache_date,
                }
                print(f"💾 快取已儲存：{cache_file}")
            except Exception as e:
                print(f"⚠️ 快取儲存失敗：{e}")
                _screener_status["cache_info"] = {
                    "from_cache": False,
                    "cache_time": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "cache_date": cache_date,
                }

        # ── Phase 4: 策略掃描（全部從 cache 讀取）──
        _screener_status["total"] = len(final_ids)
        _screener_status["current"] = f"策略掃描中 (0/{len(final_ids)})"

        def _fetch_price_cached(sid):
            return price_cache.get(sid, pd.DataFrame())

        def _fetch_inst_cached(sid):
            if sid in inst_cache:
                return inst_cache[sid]
            # TPEX 法人全市場被封時，按需逐股從 FinMind 查詢
            if exchange_map.get(sid) == "tpex":
                df = fetch_institutional_single(sid)
                if not df.empty:
                    inst_cache[sid] = df
                return df
            return pd.DataFrame()

        def _get_name_cached(sid, price_df):
            if sid in name_map:
                return name_map[sid]
            return _get_stock_name(sid, price_df)

        rows = scan_stocks(
            stock_ids=final_ids,
            strategies=strategies,
            status=_screener_status,
            fetch_price_fn=_fetch_price_cached,
            fetch_institutional_fn=_fetch_inst_cached,
            get_name_fn=_get_name_cached,
            cancel_flag={"status_ref": _screener_status},
            industry_map=industry_map,
        )

        # ── Phase 5: 補充法人買賣超 + 營收年增 ──
        _screener_status["current"] = f"補充基本面資料（{len(rows)} 檔）"
        import time as _time
        for r in rows:
            sid = r["stock_id"]
            # 法人買賣超（最近一日淨額，股→張）
            inst_df = inst_cache.get(sid, pd.DataFrame())
            if not inst_df.empty:
                latest_inst = inst_df.iloc[-1]
                fb = int(latest_inst.get("Foreign_Investor_Buy", 0))
                fs = int(latest_inst.get("Foreign_Investor_Sell", 0))
                tb = int(latest_inst.get("Investment_Trust_Buy", 0))
                ts = int(latest_inst.get("Investment_Trust_Sell", 0))
                r["foreign_net"] = (fb - fs) // 1000   # 張
                r["trust_net"] = (tb - ts) // 1000
            else:
                r["foreign_net"] = None
                r["trust_net"] = None
            # 營收年增率
            try:
                rev_df = fetch_revenue(sid)
                rev_summary = get_revenue_summary(rev_df)
                r["rev_yoy"] = rev_summary.get("年增率(%)")
            except Exception:
                r["rev_yoy"] = None
            _time.sleep(0.15)  # FinMind rate limit

        # ── Phase 6: Watchlist 追蹤（雙池系統 V2.0）──
        _screener_status["current"] = "更新追蹤清單..."
        try:
            from watchlist import update_watchlist_after_scan
            from screener import compute_screener_indicators
            rows = update_watchlist_after_scan(
                rows, cache_date, price_cache, compute_screener_indicators
            )
        except Exception as e:
            print(f"⚠️ watchlist 更新失敗（不影響結果）：{e}")

        # 排序：E-only 墊底 → 回檔買點置頂 → Pool B → Pool A → AI 優先產業 → 觸發數多
        AI_PRIORITY_INDUSTRIES = {"半導體業", "電腦及週邊設備業", "電子零組件業"}
        _pool_order = {"B": 0, "A": 1}
        rows.sort(key=lambda r: (
            1 if r.get("triggered", []) == ["E"] else 0,  # E-only 排最後
            0 if r.get("pullback_buy") else 1,  # 🎯 回檔買點置頂
            _pool_order.get(r.get("pool"), 2),
            0 if r.get("pool_transition") == "轉強訊號" else 1,
            0 if r.get("industry", "") in AI_PRIORITY_INDUSTRIES else 1,
            -len(r.get("triggered", [])),
            r.get("triggered", ["Z"]),
        ))

        _screener_status["rows"] = rows
    except Exception as e:
        _screener_status["error"] = str(e)
    finally:
        _screener_status["done"]    = True
        _screener_status["running"] = False
        _screener_status["cancel"]  = False


def _run_monitor(stock_ids: list):
    """手動觸發一次盤中量能檢查"""
    global _monitor_status

    from data_fetcher import fetch_price
    from screener import estimate_intraday_volume

    _monitor_status["running"]   = True
    _monitor_status["done"]      = False
    _monitor_status["error"]     = None
    _monitor_status["stock_ids"] = stock_ids

    try:
        now = datetime.now()
        market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
        alerts = []

        for sid in stock_ids:
            try:
                df = fetch_price(sid)
                if df.empty or len(df) < 2:
                    continue

                today_row = df.iloc[-1]
                prev_row = df.iloc[-2]

                result = estimate_intraday_volume(
                    current_volume=int(today_row["volume"]),
                    market_open=market_open,
                    now=now,
                    prev_close_volume=int(prev_row["volume"]),
                )
                alerts.append({
                    "stock_id":  sid,
                    "price":     float(today_row["close"]),
                    "vol_ratio": round(result["vol_ratio"], 2),
                    "estimated_volume": result["estimated_volume"],
                    "alert":     result["alert"],
                    "checked_at": now.strftime("%H:%M"),
                })
            except Exception as e:
                print(f"盤中監控 {sid} 失敗：{e}")

        _monitor_status["alerts"]     = alerts
        _monitor_status["last_check"] = now.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        _monitor_status["error"] = str(e)
    finally:
        _monitor_status["done"]    = True
        _monitor_status["running"] = False


# ═══════════════════════════════════════════════════════════
# 盤中即時掃描（Strategy H）
# ═══════════════════════════════════════════════════════════

def _run_intraday_scan():
    """
    盤中即時掃描：用即時報價 + 歷史快取跑 Strategy H。
    1. 讀取今日已建立的歷史快取（price_cache）
    2. 批次抓取即時報價，替換/追加今日 OHLCV
    3. 只跑 Strategy H（進場訊號）
    4. 回傳符合條件的股票 + 進出場價位
    """
    global _intraday_status
    import os, pickle

    try:
        _intraday_status["running"] = True
        _intraday_status["done"] = False
        _intraday_status["error"] = None
        _intraday_status["rows"] = []

        from data_fetcher import fetch_realtime_quotes_batch
        from screener import compute_screener_indicators
        from strategies import discover_strategies, get_strategy

        discover_strategies()
        h_info = get_strategy("H")
        if not h_info:
            _intraday_status["error"] = "Strategy H 未註冊"
            return

        # ── 1. 載入歷史快取 ──
        _intraday_status["current"] = "載入歷史資料快取..."
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        today_str = datetime.now().strftime("%Y%m%d")

        # 找最新的快取檔
        cache_file = None
        if os.path.exists(cache_dir):
            caches = sorted(
                [f for f in os.listdir(cache_dir) if f.startswith("scan_") and f.endswith(".pkl")],
                reverse=True
            )
            if caches:
                cache_file = os.path.join(cache_dir, caches[0])

        if not cache_file or not os.path.exists(cache_file):
            _intraday_status["error"] = "請先執行一次盤後掃描（建立歷史資料快取）"
            return

        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        price_cache = cached.get("price_cache", {})

        if not price_cache:
            _intraday_status["error"] = "快取中無歷史行情資料"
            return

        stock_ids = list(price_cache.keys())
        _intraday_status["total"] = len(stock_ids)
        print(f"📡 盤中掃描：{len(stock_ids)} 檔股票")

        # ── 2. 批次抓即時報價 ──
        _intraday_status["current"] = f"抓取即時報價（{len(stock_ids)} 檔）..."

        # 建立 exchange_map
        exchange_map = {}
        for sid, df in price_cache.items():
            # 猜測交易所：5開頭多為 OTC
            if sid.startswith(("3", "4", "5", "6", "7", "8")):
                exchange_map[sid] = "tpex"  # 嘗試 OTC 優先
            else:
                exchange_map[sid] = "twse"

        rt_quotes = fetch_realtime_quotes_batch(stock_ids, exchange_map={}, batch_size=50)
        print(f"   ✅ 取得 {len(rt_quotes)} 檔即時報價")

        if not rt_quotes:
            _intraday_status["error"] = "無法取得即時報價（可能非交易時間）"
            return

        # ── 3. 產業分類 ──
        from data_fetcher import fetch_industry_map
        industry_map = fetch_industry_map()

        # ── 4. 逐股更新即時資料 + 跑 Strategy H ──
        _intraday_status["current"] = "即時策略掃描中..."
        results = []
        from strategies._helpers import check_profitability

        for i, sid in enumerate(stock_ids):
            _intraday_status["progress"] = i

            if sid not in rt_quotes:
                continue

            hist_df = price_cache[sid]
            if hist_df.empty or len(hist_df) < 65:
                continue

            rt = rt_quotes[sid]
            hist_df = hist_df.copy()

            # 解析即時報價日期
            rt_date_str = rt.get("date", "")
            rt_date = None
            if rt_date_str:
                try:
                    rt_date = pd.to_datetime(rt_date_str.replace("/", ""))
                except Exception:
                    pass

            last_hist_date = hist_df["date"].iloc[-1]
            if isinstance(last_hist_date, str):
                last_hist_date = pd.to_datetime(last_hist_date)

            # 追加或替換今日即時資料
            if rt_date and rt_date > last_hist_date:
                new_row = pd.DataFrame([{
                    "date": rt_date,
                    "open": rt.get("open") or rt["price"],
                    "high": rt.get("high") or rt["price"],
                    "low": rt.get("low") or rt["price"],
                    "close": rt["price"],
                    "volume": rt.get("volume", 0),
                }])
                hist_df = pd.concat([hist_df, new_row], ignore_index=True)
            elif rt_date and rt_date == last_hist_date:
                idx = hist_df.index[-1]
                hist_df.loc[idx, "close"] = rt["price"]
                if rt.get("high"):
                    hist_df.loc[idx, "high"] = max(hist_df.loc[idx, "high"], rt["high"])
                if rt.get("low"):
                    hist_df.loc[idx, "low"] = min(hist_df.loc[idx, "low"], rt["low"])
                if rt.get("volume"):
                    hist_df.loc[idx, "volume"] = rt["volume"]

            # 計算指標 + 跑 Strategy H
            try:
                enriched = compute_screener_indicators(hist_df)
                industry = industry_map.get(sid, "")
                result = h_info.func(enriched, industry=industry)

                if result:
                    # 基本面檢查
                    profit_ok, profit_reason = check_profitability(sid)
                    if not profit_ok:
                        continue

                    last = enriched.iloc[-1]
                    prev = enriched.iloc[-2] if len(enriched) >= 2 else last
                    change_pct = round((last["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0

                    results.append({
                        "stock_id": sid,
                        "name": rt.get("name", sid),
                        "close": rt["price"],
                        "change_pct": change_pct,
                        "ma5": result.get("ma5"),
                        "ma10": result.get("ma10"),
                        "ma20": result.get("ma20"),
                        "vol_ratio": result.get("vol_ratio"),
                        "industry": industry,
                        "triggered": ["H"],
                        "strategy_labels": "朱家泓最佳⭐",
                        "strategy_details": {"H": result},
                        "profitability": profit_reason,
                        "pool": None,
                        "pullback_buy": False,
                        "pullback_steps": [],
                        "rt_time": rt.get("time", ""),
                        "is_realtime": True,
                    })
            except Exception as e:
                continue

        _intraday_status["rows"] = results
        print(f"   🎯 盤中 H 策略命中：{len(results)} 檔")

    except Exception as e:
        import traceback; traceback.print_exc()
        _intraday_status["error"] = str(e)
    finally:
        _intraday_status["done"] = True
        _intraday_status["running"] = False


# ═══════════════════════════════════════════════════════════
# 頁面 HTML
# ═══════════════════════════════════════════════════════════

_NAV_STYLE = """
<style>
:root{--bg:#0d1117;--bg2:#161b22;--bg3:#1c2128;--border:#30363d;--text:#e6edf3;--text2:#8b949e;--green:#3fb950;--red:#f85149;--blue:#58a6ff;--yellow:#e3b341;--orange:#ffa657;}
*{margin:0;padding:0;box-sizing:border-box;scrollbar-width:none;-ms-overflow-style:none;}
*::-webkit-scrollbar{display:none;}
body{background:var(--bg);color:var(--text);font-family:'Noto Sans TC',sans-serif;min-height:100vh;}
.topnav{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 24px;display:flex;align-items:center;gap:0;}
.topnav-brand{font-size:16px;font-weight:700;padding:16px 16px 16px 0;margin-right:8px;white-space:nowrap;}
.topnav a{display:inline-block;padding:16px 16px;font-size:14px;color:var(--text2);text-decoration:none;border-bottom:2px solid transparent;transition:color .2s,border-color .2s;}
.topnav a:hover{color:var(--text);}
.topnav a.active{color:var(--blue);border-bottom-color:var(--blue);}
</style>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
"""

def _nav(active: str) -> str:
    pages = [("/", "查詢"), ("/portfolio", "持股管理"), ("/batch", "批次分析"), ("/screener", "選股"), ("/chu-review", "朱家泓覆盤"), ("/h-diagnose", "H策略診斷")]
    links = "".join(
        f'<a href="{url}" class="{"active" if active==url else ""}">{label}</a>'
        for url, label in pages
    )
    return f'<nav class="topnav"><span class="topnav-brand">📊 台股覆盤</span>{links}</nav>'


# ── 單支查詢頁 ─────────────────────────────────────────────
QUERY_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>台股查詢</title>
__NAV_STYLE__
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1/dist/chartjs-chart-financial.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
<style>
/* ── 版面 ── */
.layout{display:flex;align-items:flex-start;min-height:calc(100vh - 53px);}
.sidebar{width:192px;min-width:192px;background:var(--bg2);border-right:1px solid var(--border);padding:12px 10px;display:flex;flex-direction:column;gap:6px;overflow-y:auto;max-height:calc(100vh - 53px);position:sticky;top:0;}
.sidebar-hd{display:flex;align-items:center;justify-content:space-between;padding-bottom:8px;border-bottom:1px solid var(--border);flex-shrink:0;}
.sidebar-title{font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;}
.sidebar-clear{background:transparent;border:none;color:var(--text2);font-size:10px;cursor:pointer;padding:0;font-family:'Noto Sans TC',sans-serif;}
.sidebar-clear:hover{color:var(--red);}
.sidebar-empty{font-size:12px;color:var(--text2);text-align:center;padding:16px 0;}
/* ── 側欄卡片 ── */
.s-card{background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:8px 10px;cursor:pointer;transition:border-color .2s;position:relative;}
.s-card:hover{border-color:var(--blue);}
.s-top{display:flex;align-items:flex-start;justify-content:space-between;gap:4px;margin-bottom:5px;}
.s-id{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--blue);}
.s-name{font-size:11px;color:var(--text2);margin-top:1px;}
.s-badge{border:1px solid;border-radius:10px;padding:1px 7px;font-size:10px;font-weight:600;white-space:nowrap;flex-shrink:0;}
.s-del{background:transparent;border:none;color:var(--text2);font-size:13px;cursor:pointer;padding:0 0 0 4px;line-height:1;flex-shrink:0;}
.s-del:hover{color:var(--red);}
.s-row{display:flex;justify-content:space-between;align-items:center;font-size:11px;margin-top:3px;}
.s-row-label{color:var(--text2);}
.s-row-val{font-family:'JetBrains Mono',monospace;color:var(--text);font-size:11px;}
.s-date{font-size:10px;color:var(--text2);margin-top:5px;padding-top:4px;border-top:1px solid var(--border);font-family:'JetBrains Mono',monospace;}
/* ── 主區 ── */
.main-area{flex:1;min-width:0;padding:24px;display:flex;flex-direction:column;align-items:center;gap:16px;}
.search-box{display:flex;gap:8px;width:100%;max-width:780px;}
.search-input{flex:1;background:var(--bg2);border:1px solid var(--border);border-radius:10px;padding:12px 16px;color:var(--text);font-size:16px;font-family:'JetBrains Mono',monospace;outline:none;transition:border-color .2s;}
.search-input:focus{border-color:var(--blue);}
.search-input::placeholder{color:var(--text2);}
.search-btn{background:var(--blue);color:#fff;border:none;border-radius:10px;padding:12px 24px;font-size:15px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;white-space:nowrap;}
.search-btn:hover{opacity:.85;}
.search-btn:disabled{opacity:.5;cursor:not-allowed;}
.cache-bar{display:none;max-width:780px;width:100%;background:rgba(227,179,65,.1);border:1px solid var(--yellow);border-radius:8px;padding:8px 14px;align-items:center;justify-content:space-between;gap:12px;}
.refresh-btn{background:transparent;border:1px solid var(--yellow);border-radius:6px;padding:3px 12px;font-size:11px;color:var(--yellow);cursor:pointer;font-family:'Noto Sans TC',sans-serif;white-space:nowrap;}
.loading{display:none;flex-direction:column;align-items:center;gap:12px;padding:40px;}
.spinner{width:36px;height:36px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.loading-text{font-size:13px;color:var(--text2);}
.error-msg{display:none;background:rgba(248,81,73,.1);border:1px solid var(--red);border-radius:10px;padding:14px 18px;max-width:780px;width:100%;color:var(--red);font-size:13px;text-align:center;}
.result-wrap{width:100%;max-width:780px;}
/* ── 股票卡片 ── */
.stock-card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:20px;display:flex;flex-direction:column;gap:14px;animation:fadeIn .4s ease both;}
@keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:translateY(0)}}
.card-top{display:flex;align-items:center;justify-content:space-between;}
.card-id-block{display:flex;align-items:baseline;gap:8px;}
.card-code{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600;color:var(--blue);}
.card-name{font-size:15px;font-weight:500;}
.action-badge{border:1px solid;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;}
.price-section{display:flex;flex-direction:column;gap:4px;padding-bottom:14px;border-bottom:1px solid var(--border);}
.price-main{display:flex;align-items:baseline;gap:10px;}
.current-price{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:600;}
.price-change{font-family:'JetBrains Mono',monospace;font-size:14px;}
.profit-block{font-family:'JetBrains Mono',monospace;font-size:13px;}
.shares-note{font-size:11px;color:var(--text2);margin-left:6px;}
.indicators-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.ind-block{background:var(--bg3);border-radius:8px;padding:8px;text-align:center;}
.ind-label{font-size:10px;color:var(--text2);margin-bottom:4px;}
.ind-value{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:600;line-height:1.2;}
.ind-sub{font-size:10px;color:var(--text2);margin-top:2px;}
.mini-charts{display:flex;flex-direction:column;gap:6px;background:var(--bg3);border-radius:10px;padding:12px;}
.mini-chart-wrap{display:flex;flex-direction:column;gap:3px;}
.mini-chart-label{font-size:10px;color:var(--text2);}
canvas{width:100%!important;}
.analysis-section{display:flex;flex-direction:column;gap:10px;}
.analysis-summary{font-size:13px;font-weight:500;background:var(--bg3);border-radius:8px;padding:10px 12px;line-height:1.5;}
.analysis-details{display:flex;flex-direction:column;gap:6px;}
.analysis-item{display:flex;gap:8px;font-size:12px;line-height:1.5;}
.analysis-label{color:var(--text2);white-space:nowrap;flex-shrink:0;font-size:11px;margin-top:1px;}
.action-block{border-left:3px solid;padding:10px 14px;background:var(--bg3);border-radius:0 8px 8px 0;display:flex;flex-direction:column;gap:6px;}
.action-title{font-size:13px;font-weight:700;}
.action-text{font-size:12px;line-height:1.5;}
.strategy-row{display:flex;flex-direction:column;gap:5px;margin-top:2px;padding-top:6px;border-top:1px solid var(--border);}
.strategy-item{display:flex;gap:6px;font-size:11px;line-height:1.5;}
.strategy-label{white-space:nowrap;flex-shrink:0;font-weight:600;}
.stop-loss .strategy-label{color:var(--red);}
.reversal .strategy-label{color:var(--green);}
.strategy-text{color:var(--text2);}
.risk-note{font-size:11px;color:var(--text2);}
.card-footer{display:flex;align-items:center;justify-content:space-between;padding-top:10px;border-top:1px solid var(--border);}
.footer-date{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);}
.tg-btn{background:linear-gradient(135deg,#0088cc,#229ed9);color:#fff;border:none;border-radius:8px;padding:6px 14px;font-size:12px;font-weight:600;cursor:pointer;display:flex;align-items:center;gap:5px;}
.tg-btn:hover{opacity:.85;}
.tg-btn:disabled{opacity:.5;cursor:not-allowed;}
@media(max-width:700px){
  .layout{flex-direction:column;}
  .sidebar{width:100%;min-width:0;max-height:none;position:static;flex-direction:row;flex-wrap:nowrap;overflow-x:auto;overflow-y:hidden;border-right:none;border-bottom:1px solid var(--border);padding:10px;gap:8px;}
  .sidebar-hd{display:none;}
  .s-card{min-width:155px;flex-shrink:0;}
}
@media(max-width:600px){.indicators-row{grid-template-columns:repeat(2,1fr);}}
</style>
</head>
<body>
__NAV__
<div class="layout">
  <div class="sidebar" id="sidebar">
    <div class="sidebar-hd">
      <span class="sidebar-title">最近查詢</span>
      <button class="sidebar-clear" onclick="clearAll()">清除全部</button>
    </div>
    <div id="sidebarCards"></div>
  </div>
  <div class="main-area">
    <div class="search-box">
      <input class="search-input" id="sid" type="text" placeholder="輸入代號，例如 2330" maxlength="6" inputmode="numeric" autocomplete="off">
      <button class="search-btn" id="searchBtn" onclick="doQuery(false)">查詢</button>
    </div>
    <div class="cache-bar" id="cacheBar">
      <span style="font-size:12px;color:var(--yellow)">⚡ 今日快取結果</span>
      <button class="refresh-btn" onclick="doQuery(true)">重新查詢 AI</button>
    </div>
    <div class="loading" id="loading">
      <div class="spinner"></div>
      <div class="loading-text" id="loadingText">資料抓取中...</div>
    </div>
    <div class="error-msg" id="errMsg"></div>
    <div class="result-wrap" id="result"></div>
  </div>
</div>
<script>
var rdata=JSON.parse(localStorage.getItem('rsq_rdata')||'[]');
var _memCache={};  // session 內快取：{sid: {html,name,action,analysis,signals}}
renderRecent();
document.getElementById('sid').addEventListener('keydown',function(e){if(e.key==='Enter')doQuery(false);});
// 支援 ?q=股票代號 直接查詢（從選股頁跳轉）
(function(){var p=new URLSearchParams(window.location.search).get('q');if(p){document.getElementById('sid').value=p;doQuery(false);}})();

function _showResult(sid,d,fromMemCache){
  document.getElementById('result').innerHTML=d.html;
  if(fromMemCache||d.cached)document.getElementById('cacheBar').style.display='flex';
  window._cur={stock_id:sid,name:d.name,analysis:d.analysis,signals:d.signals};
  var footer=document.querySelector('.card-footer');
  if(footer&&!footer.querySelector('.tg-btn')){
    var btn=document.createElement('button');
    btn.className='tg-btn';btn.innerHTML='📤 Telegram';
    btn.onclick=sendTg;footer.insertBefore(btn,footer.firstChild);
  }
  document.getElementById('result').querySelectorAll('script').forEach(function(s){
    var ns=document.createElement('script');ns.textContent=s.textContent;document.body.appendChild(ns);
  });
}
function doQuery(force){
  var sid=document.getElementById('sid').value.trim();
  if(!sid)return;
  // 非強制重查且記憶體有快取，直接顯示
  if(!force&&_memCache[sid]){
    document.getElementById('errMsg').style.display='none';
    document.getElementById('cacheBar').style.display='none';
    document.getElementById('result').innerHTML='';
    _showResult(sid,_memCache[sid],true);
    return;
  }
  setLoading(true);
  document.getElementById('errMsg').style.display='none';
  document.getElementById('cacheBar').style.display='none';
  document.getElementById('result').innerHTML='';
  var steps=['資料抓取中...','AI 分析中...'],si=0;
  var t=setInterval(function(){si=Math.min(si+1,steps.length-1);document.getElementById('loadingText').textContent=steps[si];},3000);
  fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stock_id:sid,force:force})})
  .then(function(r){return r.json();})
  .then(function(d){
    clearInterval(t);setLoading(false);
    if(d.error){showErr(d.error);return;}
    _memCache[sid]=d;
    addRecent(sid,d.name||sid,d);
    _showResult(sid,d,false);
  })
  .catch(function(){clearInterval(t);setLoading(false);showErr('連線失敗');});
}
function setLoading(on){document.getElementById('loading').style.display=on?'flex':'none';document.getElementById('searchBtn').disabled=on;}
function showErr(m){var e=document.getElementById('errMsg');e.textContent=m;e.style.display='block';}
function addRecent(sid,name,d){
  var now=new Date();
  var ds=now.getFullYear()+'/'+(now.getMonth()+1).toString().padStart(2,'0')+'/'+now.getDate().toString().padStart(2,'0');
  var item={sid:sid,name:name,action:(d.analysis&&d.analysis.action)||'觀望',close:(d.signals&&d.signals.close)||0,stop:(d.signals&&d.signals.ma20)||0,date:ds};
  rdata=[item].concat(rdata.filter(function(r){return r.sid!==sid;})).slice(0,12);
  localStorage.setItem('rsq_rdata',JSON.stringify(rdata));
  renderRecent();
}
function renderRecent(){
  var el=document.getElementById('sidebarCards');
  if(!rdata.length){el.innerHTML='<div class="sidebar-empty">查詢後顯示</div>';return;}
  var cm={'持有':'var(--blue)','加碼':'var(--green)','減碼':'var(--orange)','觀望':'var(--yellow)','停損':'var(--red)'};
  el.innerHTML=rdata.map(function(item,idx){
    var c=cm[item.action]||'var(--text2)';
    var h='<div class="s-card" data-sid="'+item.sid+'">';
    h+='<div class="s-top"><div><div class="s-id">'+item.sid+'</div><div class="s-name">'+item.name+'</div></div>';
    h+='<div style="display:flex;align-items:flex-start;gap:2px">';
    h+='<span class="s-badge" style="color:'+c+';border-color:'+c+'">'+item.action+'</span>';
    h+='<button class="s-del" data-del="'+idx+'">×</button>';
    h+='</div></div>';
    h+='<div class="s-row"><span class="s-row-label">買入</span><span class="s-row-val">'+item.close+'</span></div>';
    h+='<div class="s-row"><span class="s-row-label">停損</span><span class="s-row-val">'+(item.stop||'-')+'</span></div>';
    h+='<div class="s-date">'+item.date+'</div></div>';
    return h;
  }).join('');
  el.querySelectorAll('.s-card').forEach(function(card){
    card.onclick=function(e){if(!e.target.classList.contains('s-del'))quickQuery(card.dataset.sid);};
  });
  el.querySelectorAll('.s-del').forEach(function(btn){
    btn.onclick=function(e){e.stopPropagation();var i=parseInt(btn.dataset.del);rdata.splice(i,1);localStorage.setItem('rsq_rdata',JSON.stringify(rdata));renderRecent();};
  });
}
function quickQuery(s){document.getElementById('sid').value=s;doQuery(false);}
function clearAll(){if(!rdata.length)return;if(!confirm('確定清除所有查詢紀錄？'))return;rdata=[];localStorage.setItem('rsq_rdata',JSON.stringify(rdata));renderRecent();}
function sendTg(){
  if(!window._cur)return;
  var btn=event.target;btn.disabled=true;btn.textContent='傳送中...';
  fetch('/api/telegram/single',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(window._cur)})
  .then(function(r){return r.json();})
  .then(function(d){
    btn.textContent=d.success?'✓ 已傳送':'❌ 失敗';
    setTimeout(function(){btn.innerHTML='📤 Telegram';btn.disabled=false;},2000);
  }).catch(function(){btn.textContent='❌ 失敗';btn.disabled=false;});
}
</script>
</body>
</html>"""


# ── 持股管理頁 ─────────────────────────────────────────────
PORTFOLIO_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>持股管理</title>
__NAV_STYLE__
<style>
.page{padding:24px;max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:16px;}
.warn{background:rgba(227,179,65,.1);border:1px solid var(--yellow);border-radius:8px;padding:10px 14px;font-size:12px;color:var(--yellow);}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px;}
.card-title{font-size:15px;font-weight:600;}
.form-row{display:flex;gap:8px;flex-wrap:wrap;}
.form-row input{flex:1;min-width:100px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:8px 12px;color:var(--text);font-size:14px;font-family:'JetBrains Mono',monospace;outline:none;}
.form-row input:focus{border-color:var(--blue);}
.form-row input::placeholder{color:var(--text2);font-family:'Noto Sans TC',sans-serif;font-size:12px;}
.btn{border:none;border-radius:8px;padding:8px 18px;font-size:13px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;white-space:nowrap;}
.btn-blue{background:var(--blue);color:#fff;}
.btn-blue:hover{opacity:.85;}
.btn-green{background:var(--green);color:#fff;}
.btn-green:hover{opacity:.85;}
.btn-red{background:transparent;border:1px solid var(--red);color:var(--red);padding:4px 10px;font-size:12px;border-radius:6px;cursor:pointer;}
.btn-red:hover{background:rgba(248,81,73,.1);}
.btn-gray{background:var(--bg3);color:var(--text2);border:1px solid var(--border);}
.btn-gray:hover{border-color:var(--text2);color:var(--text);}
table{width:100%;border-collapse:collapse;font-size:13px;}
th{text-align:left;color:var(--text2);font-size:11px;padding:6px 8px;border-bottom:1px solid var(--border);}
td{padding:8px;border-bottom:1px solid var(--border);}
td:first-child{font-family:'JetBrains Mono',monospace;color:var(--blue);}
tr:last-child td{border-bottom:none;}
.empty{color:var(--text2);font-size:13px;text-align:center;padding:20px;}
.csv-area{display:flex;gap:8px;align-items:center;}
.msg{font-size:13px;padding:6px 0;}
.msg.ok{color:var(--green);}
.msg.err{color:var(--red);}
</style>
</head>
<body>
__NAV__
<div class="page">
  <div class="warn">⚠️ 伺服器重新啟動後資料會清空，請定期使用「匯出 CSV」備份持股清單。</div>

  <!-- 新增 / 更新 -->
  <div class="card">
    <div class="card-title">新增 / 更新持股</div>
    <div class="form-row">
      <input id="fid"    placeholder="股票代號" maxlength="6" inputmode="numeric">
      <input id="fname"  placeholder="名稱（選填）">
      <input id="fshare" placeholder="股數" type="number" min="0">
      <input id="fcost"  placeholder="成本價" type="number" min="0" step="0.01">
      <button class="btn btn-blue" onclick="upsert()">儲存</button>
    </div>
    <div class="msg" id="upsertMsg"></div>
  </div>

  <!-- 持股清單 -->
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div class="card-title">持股清單</div>
      <button class="btn btn-gray" onclick="exportCsv()">匯出 CSV</button>
    </div>
    <div id="tableWrap"></div>
  </div>

  <!-- CSV 匯入 -->
  <div class="card">
    <div class="card-title">匯入 CSV</div>
    <div style="font-size:12px;color:var(--text2);margin-bottom:4px">格式：stock_id,name,shares,cost_price（第一行為標頭，匯入後會覆蓋現有清單）</div>
    <div class="csv-area">
      <input type="file" id="csvFile" accept=".csv" style="color:var(--text2);font-size:13px;flex:1;">
      <button class="btn btn-green" onclick="importCsv()">匯入</button>
    </div>
    <div class="msg" id="importMsg"></div>
  </div>
</div>

<script>
loadTable();

function loadTable(){
  fetch('/api/portfolio').then(function(r){return r.json();}).then(function(d){
    var rows=d.portfolio||[];
    var w=document.getElementById('tableWrap');
    if(!rows.length){w.innerHTML='<div class="empty">尚無持股，請新增或匯入 CSV。</div>';return;}
    w.innerHTML='<table><thead><tr><th>代號</th><th>名稱</th><th>股數</th><th>成本</th><th></th></tr></thead><tbody>'+
      rows.map(function(r){
        return '<tr><td>'+r.stock_id+'</td><td>'+r.name+'</td><td>'+r.shares+'</td><td>'+r.cost_price+'</td>'+
          '<td><button class="btn-red" data-sid="'+r.stock_id+'">刪除</button></td></tr>';
      }).join('')+'</tbody></table>';
    w.querySelectorAll('.btn-red').forEach(function(btn){
      btn.onclick=function(){del(btn.dataset.sid);};
    });
  });
}

function upsert(){
  var id=document.getElementById('fid').value.trim();
  var name=document.getElementById('fname').value.trim();
  var shares=document.getElementById('fshare').value.trim();
  var cost=document.getElementById('fcost').value.trim();
  var msg=document.getElementById('upsertMsg');
  if(!id||!shares||!cost){msg.className='msg err';msg.textContent='股票代號、股數、成本為必填';return;}
  fetch('/api/portfolio',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_id:id,name:name||id,shares:parseFloat(shares),cost_price:parseFloat(cost)})})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.ok){msg.className='msg ok';msg.textContent='✓ 已儲存';loadTable();['fid','fname','fshare','fcost'].forEach(function(i){document.getElementById(i).value='';});}
    else{msg.className='msg err';msg.textContent=d.error||'失敗';}
    setTimeout(function(){msg.textContent='';},3000);
  });
}

function del(sid){
  if(!confirm('確定刪除 '+sid+' ?'))return;
  fetch('/api/portfolio/'+sid,{method:'DELETE'})
  .then(function(r){return r.json();})
  .then(function(d){if(d.ok)loadTable();});
}

function exportCsv(){
  window.location.href='/api/portfolio/export';
}

function importCsv(){
  var f=document.getElementById('csvFile').files[0];
  var msg=document.getElementById('importMsg');
  if(!f){msg.className='msg err';msg.textContent='請選擇 CSV 檔案';return;}
  var reader=new FileReader();
  reader.onload=function(e){
    fetch('/api/portfolio/import',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({csv:e.target.result})})
    .then(function(r){return r.json();})
    .then(function(d){
      if(d.ok){msg.className='msg ok';msg.textContent='✓ 匯入 '+d.count+' 筆';loadTable();document.getElementById('csvFile').value='';}
      else{msg.className='msg err';msg.textContent=d.error||'失敗';}
      setTimeout(function(){msg.textContent='';},4000);
    });
  };
  reader.readAsText(f,'UTF-8');
}
</script>
</body>
</html>"""


# ── 批次分析頁 ─────────────────────────────────────────────
BATCH_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>批次分析</title>
__NAV_STYLE__
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.1.1/dist/chartjs-chart-financial.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/luxon@3.4.4/build/global/luxon.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon@1.3.1/dist/chartjs-adapter-luxon.umd.min.js"></script>
<style>
.page{padding:24px;max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:16px;}
.ctrl-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap;}
.ctrl-left{display:flex;flex-direction:column;gap:4px;}
.ctrl-title{font-size:16px;font-weight:600;}
.ctrl-sub{font-size:12px;color:var(--text2);}
.ctrl-right{display:flex;gap:8px;align-items:center;}
.btn{border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;}
.btn-blue{background:var(--blue);color:#fff;}
.btn-blue:hover{opacity:.85;}
.btn-blue:disabled{opacity:.5;cursor:not-allowed;}
.btn-gray{background:var(--bg3);color:var(--text2);border:1px solid var(--border);}
.btn-gray:hover{border-color:var(--text2);color:var(--text);}
.tg-btn{background:linear-gradient(135deg,#0088cc,#229ed9);color:#fff;border:none;border-radius:8px;padding:10px 20px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;}
.tg-btn:hover{opacity:.85;}
.tg-btn:disabled{opacity:.5;cursor:not-allowed;}
.progress-card{display:none;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;flex-direction:column;gap:10px;}
.progress-bar-bg{background:var(--bg3);border-radius:99px;height:8px;overflow:hidden;}
.progress-bar{background:var(--blue);height:100%;border-radius:99px;transition:width .3s;}
.progress-text{font-size:13px;color:var(--text2);}
.result-area{display:flex;flex-direction:column;gap:16px;}
/* 批次報告卡片 - 繼承 report_generator 樣式 */
.stock-card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;padding:20px;display:flex;flex-direction:column;gap:14px;}
.card-top{display:flex;align-items:center;justify-content:space-between;}
.card-id-block{display:flex;align-items:baseline;gap:8px;}
.card-code{font-family:'JetBrains Mono',monospace;font-size:18px;font-weight:600;color:var(--blue);}
.card-name{font-size:15px;font-weight:500;}
.action-badge{border:1px solid;border-radius:20px;padding:4px 14px;font-size:13px;font-weight:600;}
.price-section{display:flex;flex-direction:column;gap:4px;padding-bottom:14px;border-bottom:1px solid var(--border);}
.price-main{display:flex;align-items:baseline;gap:10px;}
.current-price{font-family:'JetBrains Mono',monospace;font-size:28px;font-weight:600;}
.price-change{font-family:'JetBrains Mono',monospace;font-size:14px;}
.profit-block{font-family:'JetBrains Mono',monospace;font-size:13px;}
.shares-note{font-size:11px;color:var(--text2);margin-left:6px;}
.indicators-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;}
.ind-block{background:var(--bg3);border-radius:8px;padding:8px;text-align:center;}
.ind-label{font-size:10px;color:var(--text2);margin-bottom:4px;}
.ind-value{font-family:'JetBrains Mono',monospace;font-size:15px;font-weight:600;line-height:1.2;}
.ind-sub{font-size:10px;color:var(--text2);margin-top:2px;}
.mini-charts{display:flex;flex-direction:column;gap:6px;background:var(--bg3);border-radius:10px;padding:12px;}
.mini-chart-wrap{display:flex;flex-direction:column;gap:3px;}
.mini-chart-label{font-size:10px;color:var(--text2);}
canvas{width:100%!important;}
.analysis-section{display:flex;flex-direction:column;gap:10px;}
.analysis-summary{font-size:13px;font-weight:500;background:var(--bg3);border-radius:8px;padding:10px 12px;line-height:1.5;}
.analysis-details{display:flex;flex-direction:column;gap:6px;}
.analysis-item{display:flex;gap:8px;font-size:12px;line-height:1.5;}
.analysis-label{color:var(--text2);white-space:nowrap;flex-shrink:0;font-size:11px;margin-top:1px;}
.action-block{border-left:3px solid;padding:10px 14px;background:var(--bg3);border-radius:0 8px 8px 0;display:flex;flex-direction:column;gap:6px;}
.action-title{font-size:13px;font-weight:700;}
.action-text{font-size:12px;line-height:1.5;}
.strategy-row{display:flex;flex-direction:column;gap:5px;margin-top:2px;padding-top:6px;border-top:1px solid var(--border);}
.strategy-item{display:flex;gap:6px;font-size:11px;line-height:1.5;}
.strategy-label{white-space:nowrap;flex-shrink:0;font-weight:600;}
.stop-loss .strategy-label{color:var(--red);}
.reversal .strategy-label{color:var(--green);}
.strategy-text{color:var(--text2);}
.risk-note{font-size:11px;color:var(--text2);}
.card-footer{display:flex;align-items:center;justify-content:flex-end;padding-top:10px;border-top:1px solid var(--border);}
.footer-date{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--text2);}
@media(max-width:600px){.indicators-row{grid-template-columns:repeat(2,1fr);}}
</style>
</head>
<body>
__NAV__
<div class="page">
  <div class="ctrl-card">
    <div class="ctrl-left">
      <div class="ctrl-title">批次分析</div>
      <div class="ctrl-sub" id="ctrlSub">分析持股清單中所有股票</div>
    </div>
    <div class="ctrl-right">
      <button class="tg-btn" id="tgBtn" onclick="sendBatchTg()" style="display:none">📤 Telegram</button>
      <button class="btn btn-gray" id="forceBtn" onclick="runBatch(true)" style="display:none">強制重跑 AI</button>
      <button class="btn btn-blue" id="runBtn" onclick="runBatch(false)">開始分析</button>
    </div>
  </div>

  <div class="progress-card" id="progressCard">
    <div class="progress-text" id="progressText">準備中...</div>
    <div class="progress-bar-bg"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  </div>

  <div class="result-area" id="resultArea"></div>
</div>

<script>
var pollTimer=null;

function runBatch(force){
  document.getElementById('runBtn').disabled=true;
  document.getElementById('forceBtn').style.display='none';
  document.getElementById('tgBtn').style.display='none';
  document.getElementById('resultArea').innerHTML='';
  var pc=document.getElementById('progressCard');
  pc.style.display='flex';
  document.getElementById('progressText').textContent='準備中...';
  document.getElementById('progressBar').style.width='0%';

  fetch('/api/batch/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({force:force})})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){
      alert(d.error);
      document.getElementById('runBtn').disabled=false;
      pc.style.display='none';
      return;
    }
    document.getElementById('ctrlSub').textContent='共 '+d.total+' 檔，分析中...';
    pollTimer=setInterval(pollStatus,2000);
  })
  .catch(function(){document.getElementById('runBtn').disabled=false;alert('連線失敗');});
}

function pollStatus(){
  fetch('/api/batch/status').then(function(r){return r.json();}).then(function(s){
    var pct=s.total>0?Math.round(s.progress/s.total*100):0;
    document.getElementById('progressBar').style.width=pct+'%';
    var txt=s.current?('分析中：'+s.current+' ('+s.progress+'/'+s.total+')'):(s.done?'完成':'準備中...');
    document.getElementById('progressText').textContent=txt;

    if(s.done||(!s.running&&s.progress===s.total&&s.total>0)){
      clearInterval(pollTimer);
      document.getElementById('progressCard').style.display='none';
      document.getElementById('runBtn').disabled=false;
      document.getElementById('forceBtn').style.display='inline-block';
      document.getElementById('ctrlSub').textContent='最近分析完成';

      if(s.error){alert('分析失敗：'+s.error);return;}
      if(s.report_html){
        document.getElementById('tgBtn').style.display='inline-block';
        document.getElementById('resultArea').innerHTML=s.report_html;
        document.getElementById('resultArea').querySelectorAll('script').forEach(function(sc){
          var ns=document.createElement('script');ns.textContent=sc.textContent;document.body.appendChild(ns);
        });
      }
    }
  });
}

function sendBatchTg(){
  var btn=document.getElementById('tgBtn');
  btn.disabled=true;btn.textContent='傳送中...';
  fetch('/api/telegram/batch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})})
  .then(function(r){return r.json();})
  .then(function(d){
    btn.textContent=d.success?'✓ 已傳送':'❌ 失敗';
    setTimeout(function(){btn.innerHTML='📤 Telegram';btn.disabled=false;},2000);
  }).catch(function(){btn.textContent='❌ 失敗';btn.disabled=false;});
}
</script>
</body>
</html>"""


# ── 朱家泓覆盤頁 ─────────────────────────────────────────────
CHU_REVIEW_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>朱家泓覆盤</title>
__NAV_STYLE__
<style>
.page{padding:24px;max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:16px;}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px;}
.card-title{font-size:15px;font-weight:600;}
.btn{border:none;border-radius:8px;padding:10px 24px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;}
.btn-blue{background:var(--blue);color:#fff;}
.btn-blue:hover{opacity:.85;}
.btn-blue:disabled{opacity:.5;cursor:not-allowed;}
.btn-gray{background:var(--bg3);color:var(--text2);border:1px solid var(--border);}
.btn-gray:hover{border-color:var(--text2);color:var(--text);}
.ctrl-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
table{width:100%;border-collapse:collapse;font-size:13px;}
thead th{text-align:left;color:var(--text2);font-size:11px;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap;}
tbody td{padding:8px;border-bottom:1px solid var(--border);vertical-align:top;}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;white-space:nowrap;}
.badge-green{background:rgba(63,185,80,.15);color:var(--green);}
.badge-orange{background:rgba(255,166,87,.15);color:var(--orange);}
.badge-yellow{background:rgba(227,179,65,.15);color:var(--yellow);}
.badge-blue{background:rgba(88,166,255,.15);color:var(--blue);}
.badge-red{background:rgba(248,81,73,.15);color:var(--red);}
.badge-gray{background:var(--bg3);color:var(--text2);}
.td-num{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;}
/* summary */
.summary-card{display:flex;gap:16px;flex-wrap:wrap;align-items:center;}
.summary-item{display:flex;flex-direction:column;align-items:center;gap:2px;min-width:60px;}
.summary-num{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.summary-label{font-size:11px;color:var(--text2);}
.summary-num.fire{color:#ff6400;}
.summary-num.green{color:var(--green);}
.summary-num.orange{color:var(--orange);}
.summary-num.yellow{color:var(--yellow);}
.summary-num.blue{color:var(--blue);}
/* detail panel */
.detail-panel{display:none;background:var(--bg3);border-radius:8px;padding:16px;margin-top:8px;}
.detail-panel.show{display:block;}
.detail-section{margin-bottom:12px;}
.detail-section:last-child{margin-bottom:0;}
.detail-title{font-size:12px;font-weight:600;color:var(--text2);margin-bottom:6px;}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px;}
.detail-item{font-size:12px;display:flex;justify-content:space-between;gap:8px;padding:4px 8px;background:var(--bg2);border-radius:4px;}
.detail-item .label{color:var(--text2);}
.detail-item .value{font-family:'JetBrains Mono',monospace;font-weight:600;}
.signal-tag{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;margin:2px;}
.signal-buy-point{background:rgba(255,100,0,.2);color:#ff6400;font-weight:700;font-size:13px;}
.signal-reduce{background:rgba(255,166,87,.15);color:var(--orange);}
.signal-alert{background:rgba(227,179,65,.15);color:var(--yellow);}
.signal-take-profit{background:rgba(88,166,255,.15);color:var(--blue);}
.clickable-row{cursor:pointer;transition:background .15s;}
.clickable-row:hover{background:var(--bg3);}
.badge-blue-link{background:rgba(88,166,255,.15);color:var(--blue);cursor:pointer;text-decoration:none;display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;}
.badge-blue-link:hover{background:rgba(88,166,255,.3);}
/* ── 產業報告 Modal ── */
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:1000;justify-content:center;align-items:center;}
.modal-overlay.active{display:flex;}
.modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:16px;width:92%;max-width:820px;max-height:85vh;overflow-y:auto;padding:32px;position:relative;scrollbar-width:none;-ms-overflow-style:none;}
.modal-box::-webkit-scrollbar{display:none;}
.modal-close{position:absolute;top:12px;right:16px;background:none;border:none;color:var(--text2);font-size:28px;cursor:pointer;line-height:1;padding:4px 8px;border-radius:8px;}
.modal-close:hover{background:var(--bg3);color:var(--text);}
.modal-title{font-size:18px;font-weight:700;color:var(--text);margin-bottom:20px;padding-right:40px;}
.modal-section{margin-bottom:24px;}
.modal-section h3{font-size:15px;font-weight:700;color:var(--blue);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border);}
.modal-section p,.modal-section ul,.modal-section li{font-size:13px;line-height:1.7;color:var(--text);}
.modal-footer{font-size:11px;color:var(--text2);padding-top:12px;border-top:1px solid var(--border);}
.modal-loading{text-align:center;padding:40px;color:var(--text2);font-size:14px;}
.spinner{width:32px;height:32px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto;}
@keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
__NAV__
<div class="page">
  <div class="card">
    <div class="card-title">朱家泓老師 — 每日持股覆盤</div>
    <div style="font-size:12px;color:var(--text2);line-height:1.6;">
      依據朱家泓老師的操作紀律，逐檔檢查持股的出場訊號：<br>
      <b>減碼</b>：收盤跌破 5 日線 ｜ <b>警覺</b>：高位長黑K / 放量十字星 ｜ <b>落袋</b>：5 日線走平
    </div>
    <div class="ctrl-row">
      <textarea id="stockInput" rows="2" style="flex:1;min-width:200px;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg3);color:var(--text);font-size:13px;font-family:'Noto Sans TC',sans-serif;resize:vertical;" placeholder="輸入股票代號，逗號分隔：2330,2454,3008"></textarea>
    </div>
    <div class="ctrl-row">
      <button class="btn btn-gray" onclick="loadFromPortfolio()">從持股帶入</button>
      <button class="btn btn-blue" id="reviewBtn" onclick="runReview()">開始覆盤</button>
      <span style="font-size:12px;color:var(--text2);" id="statusNote"></span>
    </div>
  </div>

  <div class="card" id="summaryCard" style="display:none">
    <div class="card-title">覆盤摘要</div>
    <div class="summary-card" id="summaryArea"></div>
  </div>

  <div id="resultArea"></div>
</div>
<script>
function loadFromPortfolio(){
  document.getElementById('statusNote').textContent='載入持股中...';
  fetch('/api/portfolio')
  .then(function(r){return r.json();})
  .then(function(d){
    var ids=(d.portfolio||[]).map(function(p){return p.stock_id;});
    if(!ids.length){
      document.getElementById('statusNote').textContent='持股清單為空，請先到持股管理新增';
      return;
    }
    document.getElementById('stockInput').value=ids.join(',');
    document.getElementById('statusNote').textContent=ids.length+'檔持股已帶入';
  })
  .catch(function(e){
    document.getElementById('statusNote').textContent='載入失敗：'+e.message;
  });
}

var reviewPollTimer=null;
function runReview(){
  var raw=document.getElementById('stockInput').value.trim();
  if(!raw){alert('請輸入股票代號');return;}
  var ids=raw.split(/[,，\\s]+/).filter(Boolean);
  if(!ids.length){alert('請輸入至少一個股票代號');return;}

  document.getElementById('reviewBtn').disabled=true;
  document.getElementById('statusNote').textContent='覆盤啟動中...';
  document.getElementById('summaryCard').style.display='none';
  document.getElementById('resultArea').innerHTML='';

  fetch('/api/chu-review/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_ids:ids})
  })
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){document.getElementById('reviewBtn').disabled=false;document.getElementById('statusNote').textContent=d.error;return;}
    reviewPollTimer=setInterval(pollReview,2000);
  })
  .catch(function(e){document.getElementById('reviewBtn').disabled=false;document.getElementById('statusNote').textContent='啟動失敗：'+e;});
}
function pollReview(){
  fetch('/api/chu-review/status').then(function(r){return r.json();}).then(function(s){
    document.getElementById('statusNote').textContent=s.current||'覆盤中...';
    if(!s.done) return;
    clearInterval(reviewPollTimer);
    document.getElementById('reviewBtn').disabled=false;
    if(s.error){document.getElementById('statusNote').textContent='覆盤失敗：'+s.error;return;}
    var d=s.result;
    var intraTag=d.is_intraday?' <span style="color:#10b981;font-size:12px;">🟢 即時盤價</span>':' <span style="color:var(--text2);font-size:12px;">📦 收盤資料</span>';
    document.getElementById('statusNote').innerHTML='覆盤完成 ('+d.date+')'+intraTag;
    renderSummary(d.summary);
    renderResults(d.reviews);
  }).catch(function(){});
}

function renderSummary(s){
  if(!s)return;
  document.getElementById('summaryCard').style.display='flex';
  var items=[
    {label:'持股總數',value:s.total,cls:''},
    {label:'🎯 回檔買點',value:s.buy_point||0,cls:'fire'},
    {label:'健康續抱',value:s.healthy,cls:'green'},
    {label:'建議減碼',value:s.reduce,cls:'orange'},
    {label:'提高警覺',value:s.alert,cls:'yellow'},
    {label:'建議落袋',value:s.take_profit,cls:'blue'},
  ];
  document.getElementById('summaryArea').innerHTML=items.map(function(it){
    return '<div class="summary-item"><span class="summary-num '+it.cls+'">'+it.value+'</span><span class="summary-label">'+it.label+'</span></div>';
  }).join('');
}

var STATUS_CFG={
  buy_point:{badge:'badge-fire',icon:'🎯',label:'買點'},
  healthy:{badge:'badge-green',icon:'●',label:'健康'},
  reduce:{badge:'badge-orange',icon:'▼',label:'減碼'},
  alert:{badge:'badge-yellow',icon:'⚠',label:'警覺'},
  take_profit:{badge:'badge-blue',icon:'◆',label:'落袋'},
};
var STATUS_ORDER={buy_point:-1,reduce:0,alert:1,take_profit:2,healthy:3};

var _chuReviews=[];
var _chuSortCol='';
var _chuSortAsc=true;

var CHU_COLS=[
  {key:'stock_id',  label:'代號'},
  {key:'name',      label:'名稱'},
  {key:'close',     label:'收盤'},
  {key:'change_pct',label:'漲跌%'},
  {key:'status',    label:'狀態'},
  {key:'summary',   label:'訊號摘要'},
  {key:'alignment', label:'均線排列'},
  {key:'deduct',    label:'MA20扣抵'},
];

function renderResults(reviews){
  if(!reviews||!reviews.length)return;
  // 預設排序：buy_point 置頂，其餘按狀態嚴重度
  reviews.sort(function(a,b){
    var sa=(a.review||{}).status||'healthy';
    var sb=(b.review||{}).status||'healthy';
    var oa=STATUS_ORDER[sa]!=null?STATUS_ORDER[sa]:99;
    var ob=STATUS_ORDER[sb]!=null?STATUS_ORDER[sb]:99;
    return oa-ob;
  });
  _chuReviews=reviews;
  _renderChuTable();
}

function _renderChuTable(){
  var reviews=_chuReviews;
  var html='<div class="card"><div class="card-title">個股覆盤結果 <span style="font-size:11px;color:var(--text2);font-weight:400;">（點擊標頭排序，點擊個股展開詳情）</span></div><table><thead><tr>';
  html+=CHU_COLS.map(function(c){
    var arrow='';
    if(_chuSortCol===c.key) arrow='<span style="margin-left:3px;">'+(_chuSortAsc?'▲':'▼')+'</span>';
    return '<th style="cursor:pointer;user-select:none;" onclick="chuSortBy(&quot;'+c.key+'&quot;)">'+c.label+arrow+'</th>';
  }).join('');
  html+='</tr></thead><tbody>';

  reviews.forEach(function(r,idx){
    var rev=r.review||{};
    var st=rev.status||'healthy';
    var cfg=STATUS_CFG[st]||STATUS_CFG.healthy;
    var chg=r.change_pct||0;
    var chgCls=chg>0?'badge-green':chg<0?'badge-red':'badge-gray';

    html+='<tr class="clickable-row" onclick="toggleDetail('+idx+')">';
    html+='<td style="font-weight:600;">'+r.stock_id+'</td>';
    html+='<td>'+r.name+'</td>';
    html+='<td class="td-num">'+(r.close?r.close.toFixed(2):'-')+'</td>';
    html+='<td><span class="badge '+chgCls+'">'+(chg>0?'+':'')+chg.toFixed(2)+'%</span></td>';
    html+='<td><span class="badge '+cfg.badge+'">'+cfg.icon+' '+cfg.label+'</span></td>';
    html+='<td style="font-size:12px;">'+rev.summary+'</td>';

    var ma=rev.ma_status||{};
    html+='<td><span class="badge '+(ma.alignment==='四線多排'?'badge-green':'badge-gray')+'">'+
      (ma.alignment||'-')+'</span></td>';

    var deductOk=ma.ma20_deduct_bullish;
    html+='<td><span class="badge '+(deductOk?'badge-green':'badge-red')+'">'+(
      ma.ma20_deduct_value!=null?ma.ma20_deduct_value.toFixed(1)+(deductOk?' ↑':' ↓'):'-'
    )+'</span></td>';
    html+='</tr>';

    // detail panel
    html+='<tr id="detailRow'+idx+'"><td colspan="8" style="padding:0;border:none;">';
    html+='<div class="detail-panel" id="detail'+idx+'">';
    html+=renderDetail(r, rev);
    html+='</div></td></tr>';
  });

  html+='</tbody></table></div>';
  document.getElementById('resultArea').innerHTML=html;
}

function chuSortBy(col){
  if(_chuSortCol===col){_chuSortAsc=!_chuSortAsc;}else{_chuSortCol=col;_chuSortAsc=true;}
  _chuReviews.sort(function(a,b){
    var av,bv;
    if(col==='status'){
      av=STATUS_ORDER[(a.review||{}).status||'healthy'];
      bv=STATUS_ORDER[(b.review||{}).status||'healthy'];
    }else if(col==='summary'){
      av=(a.review||{}).summary||'';
      bv=(b.review||{}).summary||'';
    }else if(col==='alignment'){
      av=((a.review||{}).ma_status||{}).alignment||'';
      bv=((b.review||{}).ma_status||{}).alignment||'';
    }else if(col==='deduct'){
      av=((a.review||{}).ma_status||{}).ma20_deduct_value||0;
      bv=((b.review||{}).ma_status||{}).ma20_deduct_value||0;
    }else{
      av=a[col]; bv=b[col];
    }
    if(av==null) av='';
    if(bv==null) bv='';
    if(typeof av==='string') return _chuSortAsc?av.localeCompare(bv):bv.localeCompare(av);
    return _chuSortAsc?(av-bv):(bv-av);
  });
  _renderChuTable();
}

function renderDetail(r, rev){
  var html='';
  var ma=rev.ma_status||{};
  var kb=rev.k_bar||{};

  // MA detail
  html+='<div class="detail-section"><div class="detail-title">均線狀態</div><div class="detail-grid">';
  ['ma5','ma10','ma20','ma60'].forEach(function(k){
    var v=ma[k];
    html+='<div class="detail-item"><span class="label">'+k.toUpperCase()+'</span><span class="value">'+(v!=null?v.toFixed(2):'-')+'</span></div>';
  });
  html+='<div class="detail-item"><span class="label">MA5斜率</span><span class="value">'+(ma.ma5_slope!=null?(ma.ma5_slope>0?'+':'')+ma.ma5_slope.toFixed(2):'-')+'</span></div>';
  html+='<div class="detail-item"><span class="label">MA20斜率</span><span class="value">'+(ma.ma20_slope!=null?(ma.ma20_slope>0?'+':'')+ma.ma20_slope.toFixed(2):'-')+'</span></div>';
  html+='<div class="detail-item"><span class="label">MA20扣抵值</span><span class="value">'+(ma.ma20_deduct_value!=null?ma.ma20_deduct_value.toFixed(2):'-')+'</span></div>';
  html+='<div class="detail-item"><span class="label">扣抵差距</span><span class="value">'+(ma.ma20_deduct_diff_pct!=null?(ma.ma20_deduct_diff_pct>0?'+':'')+ma.ma20_deduct_diff_pct.toFixed(1)+'%':'-')+'</span></div>';
  html+='</div></div>';

  // K bar detail
  html+='<div class="detail-section"><div class="detail-title">K棒分析</div><div class="detail-grid">';
  html+='<div class="detail-item"><span class="label">實體比</span><span class="value">'+(kb.body_pct!=null?kb.body_pct.toFixed(2):'-')+'%</span></div>';
  html+='<div class="detail-item"><span class="label">上影線</span><span class="value">'+(kb.upper_shadow_pct!=null?kb.upper_shadow_pct.toFixed(2):'-')+'%</span></div>';
  html+='<div class="detail-item"><span class="label">下影線</span><span class="value">'+(kb.lower_shadow_pct!=null?kb.lower_shadow_pct.toFixed(2):'-')+'%</span></div>';
  html+='<div class="detail-item"><span class="label">位階</span><span class="value">'+(kb.position_in_range!=null?kb.position_in_range.toFixed(0):'-')+'%</span></div>';
  html+='<div class="detail-item"><span class="label">長黑K</span><span class="value">'+(kb.is_long_black?'是':'否')+'</span></div>';
  html+='<div class="detail-item"><span class="label">十字星</span><span class="value">'+(kb.is_doji?'是':'否')+'</span></div>';
  html+='</div></div>';

  // signals
  var signals=rev.signals||[];
  if(signals.length){
    html+='<div class="detail-section"><div class="detail-title">觸發訊號</div><div>';
    signals.forEach(function(s){
      var cls='signal-'+s.type.replace('_','-');
      html+='<div class="signal-tag '+cls+'">'+s.rule+'：'+s.detail+'</div>';
    });
    html+='</div></div>';
  }

  // 產業報告連結
  var nameEsc=(r.name||'').replace(/'/g,"\\\\'");
  var indEsc=(r.industry||'').replace(/'/g,"\\\\'");
  html+='<div class="detail-section"><a href="#" class="badge-blue-link" onclick="openReport(\\''+r.stock_id+'\\',\\''+nameEsc+'\\',\\''+indEsc+'\\',event)">📊 產業報告</a></div>';

  return html;
}

function toggleDetail(idx){
  var el=document.getElementById('detail'+idx);
  el.classList.toggle('show');
}

// ── 產業報告 Modal ──────────────────────────────────────────
var _reportCache={};

function openReport(sid, name, industry, evt){
  if(evt)evt.preventDefault();
  if(evt)evt.stopPropagation();
  var modal=document.getElementById('reportModal');
  var content=document.getElementById('reportContent');
  modal.classList.add('active');
  document.body.style.overflow='hidden';

  if(_reportCache[sid]){
    _renderReport(_reportCache[sid]);
    return;
  }

  content.innerHTML='<div class="modal-loading"><div class="spinner"></div><br>AI 正在生成 <b>'+name+'</b> 產業分析報告...<br><span style="font-size:12px;color:var(--text2);">首次生成約需 10-15 秒，之後走快取</span></div>';

  fetch('/api/industry-report',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_id:sid, name:name, industry:industry})
  })
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){
      content.innerHTML='<div class="modal-loading" style="color:var(--red);">❌ '+d.error+'</div>';
      return;
    }
    _reportCache[sid]=d;
    _renderReport(d);
  })
  .catch(function(e){
    content.innerHTML='<div class="modal-loading" style="color:var(--red);">❌ 報告生成失敗：'+e+'</div>';
  });
}

function _renderReport(d){
  var content=document.getElementById('reportContent');
  var s=d.sections||{};
  var html='<div class="modal-title">'+d.ticker+' '+d.name+'：產業地位與展望分析報告</div>';
  if(d.error_hint){
    html+='<div style="background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:12px;color:var(--red);">⚠️ '+d.error_hint+'</div>';
  }
  html+='<div class="modal-section"><h3>1. 產業定位與核心業務</h3>'+(s.positioning||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>2. 產業展望與利多題材</h3>'+(s.growth||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>3. 同類型競爭對手</h3>'+(s.peers||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>4. 法說會資訊</h3>'+(s.conference||'<p>無資料</p>')+'</div>';
  if(d.generated_at){
    html+='<div class="modal-footer">報告生成：'+d.generated_at+(d.cached?' ✅ 快取命中（7 天有效）':' 🤖 AI 即時生成')+'</div>';
  }
  content.innerHTML=html;
}

function closeReportModal(evt){
  if(!evt||evt.target.classList.contains('modal-overlay')){
    document.getElementById('reportModal').classList.remove('active');
    document.body.style.overflow='';
  }
}

document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){closeReportModal();}
});
</script>
<!-- 產業報告 Modal -->
<div class="modal-overlay" id="reportModal" onclick="closeReportModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeReportModal()">&times;</button>
    <div id="reportContent">
      <div class="modal-loading"><div class="spinner"></div><br>AI 產業報告生成中...</div>
    </div>
  </div>
</div>
</body>
</html>
"""


# ── H策略即時診斷頁 ──────────────────────────────────────────
H_DIAGNOSE_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>H策略診斷</title>
__NAV_STYLE__
<style>
.page{padding:24px;max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:16px;}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px;}
.card-title{font-size:15px;font-weight:600;}
.btn{border:none;border-radius:8px;padding:10px 24px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;}
.btn-blue{background:var(--blue);color:#fff;}
.btn-blue:hover{opacity:.85;}
.btn-blue:disabled{opacity:.5;cursor:not-allowed;}
.btn-gray{background:var(--bg3);color:var(--text2);border:1px solid var(--border);}
.btn-gray:hover{border-color:var(--text2);color:var(--text);}
.ctrl-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:600;white-space:nowrap;}
.badge-green{background:rgba(63,185,80,.15);color:var(--green);}
.badge-red{background:rgba(248,81,73,.15);color:var(--red);}
.badge-orange{background:rgba(255,166,87,.15);color:var(--orange);}
.badge-gray{background:var(--bg3);color:var(--text2);}
.summary-card{display:flex;gap:16px;flex-wrap:wrap;align-items:center;}
.summary-item{display:flex;flex-direction:column;align-items:center;gap:2px;min-width:60px;}
.summary-num{font-size:28px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.summary-label{font-size:11px;color:var(--text2);}
.summary-num.green{color:var(--green);}
.summary-num.orange{color:var(--orange);}
.result-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;}
.result-summary{font-size:13px;color:var(--text2);margin-bottom:12px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
thead th{text-align:left;color:var(--text2);font-size:11px;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none;}
thead th:hover{color:var(--text);}
tbody td{padding:8px;border-bottom:1px solid var(--border);vertical-align:top;}
.td-num{text-align:right;font-family:'JetBrains Mono',monospace;font-size:12px;}
.td-code a{color:var(--blue);text-decoration:none;font-weight:600;font-family:'JetBrains Mono',monospace;font-size:12px;}
.td-code a:hover{text-decoration:underline;}
.score-all{color:var(--green);font-weight:700;}
.score-most{color:var(--orange);font-weight:700;}
.score-few{color:var(--red);font-weight:700;}
.sort-arrow{display:inline-block;width:12px;font-size:9px;color:var(--text2);}
.sort-arrow.asc::after{content:'\\25B2';}
.sort-arrow.desc::after{content:'\\25BC';}
.spinner{width:32px;height:32px;border:3px solid var(--border);border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite;margin:0 auto;}
@keyframes spin{to{transform:rotate(360deg);}}
</style>
</head>
<body>
__NAV__
<div class="page">
  <div class="card">
    <div class="card-title">Strategy H 即時診斷 — 朱家泓最佳進場條件</div>
    <div style="font-size:12px;color:var(--text2);line-height:1.6;">
      9 項進場條件逐條檢查：ADX 趨勢、DI 方向、RSI 動能、四線多排、斜率、MA20 扣抵、站上 MA5、放量確認、乖離率。<br>
      使用即時盤價（盤中 9:00-13:30）或最新收盤價進行診斷。
    </div>
    <div class="ctrl-row">
      <textarea id="stockInput" rows="2" style="flex:1;min-width:200px;padding:8px;border-radius:8px;border:1px solid var(--border);background:var(--bg3);color:var(--text);font-size:13px;font-family:'Noto Sans TC',sans-serif;resize:vertical;"
        placeholder="輸入股票代號，逗號分隔：2330,2454,3008"></textarea>
    </div>
    <div class="ctrl-row">
      <button class="btn btn-gray" onclick="loadFromPortfolio()">從持股帶入</button>
      <button class="btn btn-blue" id="diagnoseBtn" onclick="runDiagnose()">開始診斷</button>
      <span style="font-size:12px;color:var(--text2);" id="statusNote"></span>
    </div>
  </div>

  <div class="card" id="overviewCard" style="display:none">
    <div class="card-title">診斷總覽</div>
    <div class="summary-card" id="overviewArea"></div>
  </div>

  <div id="resultArea"></div>
</div>

<script>
function loadFromPortfolio(){
  document.getElementById('statusNote').textContent='載入持股中...';
  fetch('/api/portfolio')
  .then(function(r){return r.json();})
  .then(function(d){
    var ids=(d.portfolio||[]).map(function(p){return p.stock_id;});
    if(!ids.length){document.getElementById('statusNote').textContent='持股清單為空';return;}
    document.getElementById('stockInput').value=ids.join(',');
    document.getElementById('statusNote').textContent=ids.length+'檔持股已帶入';
  })
  .catch(function(e){document.getElementById('statusNote').textContent='載入失敗：'+e.message;});
}

var diagPollTimer=null;
var _allResults=[];
var _sortCol='passed_count';
var _sortAsc=false;

function runDiagnose(){
  var raw=document.getElementById('stockInput').value.trim();
  if(!raw){alert('請輸入股票代號');return;}
  var ids=raw.split(/[,\\uff0c\\s]+/).filter(Boolean);
  if(!ids.length){alert('請輸入至少一個股票代號');return;}

  document.getElementById('diagnoseBtn').disabled=true;
  document.getElementById('statusNote').textContent='診斷啟動中...';
  document.getElementById('overviewCard').style.display='none';
  document.getElementById('resultArea').innerHTML='';

  fetch('/api/h-diagnose/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_ids:ids})
  })
  .then(function(r){
    if(!r.ok) return r.text().then(function(t){throw new Error(t);});
    return r.json();
  })
  .then(function(d){
    if(d.error){
      document.getElementById('diagnoseBtn').disabled=false;
      document.getElementById('statusNote').textContent=d.error;
      return;
    }
    diagPollTimer=setInterval(pollDiagnose,2000);
  })
  .catch(function(e){
    document.getElementById('diagnoseBtn').disabled=false;
    document.getElementById('statusNote').textContent='啟動失敗：'+e;
  });
}

function pollDiagnose(){
  fetch('/api/h-diagnose/status')
  .then(function(r){return r.json();})
  .then(function(s){
    document.getElementById('statusNote').textContent=s.current||'診斷中...';
    if(!s.done) return;
    clearInterval(diagPollTimer);
    document.getElementById('diagnoseBtn').disabled=false;
    if(s.error){
      document.getElementById('statusNote').textContent='診斷失敗：'+s.error;
      return;
    }
    var d=s.result;
    var intraTag=d.is_intraday
      ?' <span style="color:#10b981;font-size:12px;">\\u2022 即時盤價</span>'
      :' <span style="color:var(--text2);font-size:12px;">\\u2022 收盤資料</span>';
    document.getElementById('statusNote').innerHTML='診斷完成 ('+d.date+')'+intraTag;
    _allResults=d.results||[];
    renderOverview(_allResults);
    sortBy('passed_count');
  })
  .catch(function(){});
}

function renderOverview(results){
  if(!results||!results.length) return;
  document.getElementById('overviewCard').style.display='flex';
  var total=results.length, allPass=0, most=0, few=0;
  results.forEach(function(r){
    var d=r.diagnose||{};
    if(d.error){few++;return;}
    if(d.passed) allPass++;
    else if((d.passed_count||0)>=7) most++;
    else few++;
  });
  document.getElementById('overviewArea').innerHTML=
    '<div class="summary-item"><span class="summary-num">'+total+'</span><span class="summary-label">診斷總數</span></div>'+
    '<div class="summary-item"><span class="summary-num green">'+allPass+'</span><span class="summary-label">全數通過 (9/9)</span></div>'+
    '<div class="summary-item"><span class="summary-num orange">'+most+'</span><span class="summary-label">接近通過 (7-8)</span></div>'+
    '<div class="summary-item"><span class="summary-num" style="color:var(--red);">'+few+'</span><span class="summary-label">未達標 (&lt;7)</span></div>';
}

var COLS=[
  {key:'stock_id',    label:'代號'},
  {key:'name',        label:'名稱'},
  {key:'close',       label:'收盤'},
  {key:'passed_count',label:'通過'},
  {key:'adx',         label:'ADX(8)'},
  {key:'rsi14',       label:'RSI(14)'},
  {key:'ma_align',    label:'均線排列'},
  {key:'vol_ok',      label:'量能'},
];

function sortBy(col){
  if(_sortCol===col){_sortAsc=!_sortAsc;}else{_sortCol=col;_sortAsc=(col==='stock_id'||col==='name');}
  var sorted=_allResults.slice().sort(function(a,b){
    var da=a.diagnose||{}, db=b.diagnose||{};
    var sma=da.summary||{}, smb=db.summary||{};
    var av,bv;
    if(col==='passed_count'){av=da.passed_count||0;bv=db.passed_count||0;}
    else if(col==='adx'){av=sma.adx;bv=smb.adx;}
    else if(col==='rsi14'){av=sma.rsi14;bv=smb.rsi14;}
    else if(col==='ma_align'){av=da.passed_count||0;bv=db.passed_count||0;}
    else if(col==='vol_ok'){av=sma.volume||0;bv=smb.volume||0;}
    else{av=a[col];bv=b[col];}
    if(av==null)return 1;if(bv==null)return -1;
    if(typeof av==='string')return _sortAsc?av.localeCompare(bv):bv.localeCompare(av);
    return _sortAsc?av-bv:bv-av;
  });
  renderTable(sorted);
}

function renderTable(rows){
  var wrap=document.getElementById('resultArea');
  if(!rows.length){wrap.innerHTML='<div class="result-card"><div style="padding:16px;color:var(--text2);">無診斷結果</div></div>';return;}

  var header=COLS.map(function(c){
    var arrow='<span class="sort-arrow'+(_sortCol===c.key?(' '+(_sortAsc?'asc':'desc')):'')+'"></span>';
    return '<th onclick="sortBy(\\''+c.key+'\\')">'+c.label+arrow+'</th>';
  }).join('');

  var body=rows.map(function(r){
    var d=r.diagnose||{};
    var sm=d.summary||{};

    if(d.error){
      return '<tr><td class="td-code">'+r.stock_id+'</td><td>'+r.name+'</td><td colspan="6"><span class="badge badge-red">'+d.error+'</span></td></tr>';
    }

    var pc=d.passed_count||0;
    var tc=d.total_checks||9;
    var scoreCls=pc===tc?'score-all':(pc>=7?'score-most':'score-few');
    var scoreLabel=pc===tc?'\\u2705 '+pc+'/'+tc:pc+'/'+tc;

    // ADX cell
    var adxVal=sm.adx!=null?sm.adx.toFixed(1):'-';
    var diVal=sm.plus_di!=null&&sm.minus_di!=null?'(+'+sm.plus_di.toFixed(0)+' / -'+sm.minus_di.toFixed(0)+')':'';
    // RSI cell
    var rsiVal=sm.rsi14!=null?sm.rsi14.toFixed(1):'-';
    // MA alignment
    var maChecks=(d.checks||[]).filter(function(c){return c.name.indexOf('四線')>=0||c.name.indexOf('斜率')>=0;});
    var maOk=maChecks.every(function(c){return c.passed;});
    var maBadge=maOk?'<span class="badge badge-green">\\u2713 多排</span>':'<span class="badge badge-red">\\u2717 未排</span>';
    // Volume
    var volCheck=(d.checks||[]).find(function(c){return c.name.indexOf('量')>=0;});
    var volBadge=volCheck?(volCheck.passed?'<span class="badge badge-green">\\u2713 放量</span>':'<span class="badge badge-gray">\\u2717 量不足</span>'):'<span class="badge badge-gray">-</span>';

    var rtTag=r.rt_time?'<span style="font-size:10px;color:var(--text2);margin-left:4px;">'+r.rt_time+'</span>':'';

    var row1='<tr>'+
      '<td class="td-code"><a href="/?q='+r.stock_id+'" target="_blank">'+r.stock_id+'</a></td>'+
      '<td>'+r.name+'</td>'+
      '<td class="td-num">'+(r.close!=null?r.close.toFixed(2):'-')+rtTag+'</td>'+
      '<td class="td-num"><span class="'+scoreCls+'">'+scoreLabel+'</span></td>'+
      '<td class="td-num">'+adxVal+' <span style="font-size:10px;color:var(--text2)">'+diVal+'</span></td>'+
      '<td class="td-num">'+rsiVal+'</td>'+
      '<td>'+maBadge+'</td>'+
      '<td>'+volBadge+'</td>'+
    '</tr>';

    // Row 2: failing checks as badges
    var failChecks=(d.checks||[]).filter(function(c){return !c.passed;});
    var row2='';
    if(failChecks.length>0&&failChecks.length<9){
      var badges=failChecks.map(function(c){
        return '<span class="badge badge-red" style="font-size:10px;">\\u2717 '+c.name+' '+c.value+' (需'+c.threshold+')</span>';
      }).join(' ');
      row2='<tr><td colspan="2" style="border-top:none;padding:0;"></td>'+
        '<td colspan="6" style="border-top:none;padding-top:0;padding-bottom:10px;">'+
        '<div style="display:flex;gap:4px;flex-wrap:wrap;">'+badges+'</div></td></tr>';
    }

    return row1+row2;
  }).join('');

  wrap.innerHTML='<div class="result-card">'+
    '<div class="result-summary">診斷結果：<strong>'+rows.length+'</strong> 檔（點欄位標題排序）</div>'+
    '<div style="overflow-x:auto"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'+
  '</div>';
}
</script>
</body>
</html>
"""


# ── 選股頁 ─────────────────────────────────────────────────
SCREENER_PAGE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>選股</title>
__NAV_STYLE__
<style>
.page{padding:24px;max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:16px;}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:12px;}
.card-title{font-size:15px;font-weight:600;}
.btn{border:none;border-radius:8px;padding:10px 24px;font-size:14px;font-weight:600;cursor:pointer;font-family:'Noto Sans TC',sans-serif;}
.btn-sm{padding:7px 16px;font-size:13px;}
.btn-blue{background:var(--blue);color:#fff;}
.btn-blue:hover{opacity:.85;}
.btn-blue:disabled{opacity:.5;cursor:not-allowed;}
.btn-gray{background:var(--bg3);color:var(--text2);border:1px solid var(--border);}
.btn-gray:hover{border-color:var(--text2);color:var(--text);}
.ctrl-row{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.progress-card{display:none;background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;flex-direction:column;gap:10px;}
.progress-bar-bg{background:var(--bg3);border-radius:99px;height:8px;overflow:hidden;}
.progress-bar{background:var(--blue);height:100%;border-radius:99px;transition:width .3s;}
.progress-text{font-size:13px;color:var(--text2);}
/* strategy grid */
.strategy-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}
.strategy-checkbox{display:flex;align-items:center;gap:10px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;padding:12px;cursor:pointer;transition:border-color .2s;}
.strategy-checkbox:hover{border-color:var(--blue);}
.strategy-checkbox input[type=checkbox]{width:16px;height:16px;cursor:pointer;accent-color:var(--blue);}
.strat-badge{border-radius:4px;padding:2px 8px;font-size:12px;font-weight:700;font-family:'JetBrains Mono',monospace;}
.strat-a{background:rgba(63,185,80,.2);color:var(--green);}
.strat-b{background:rgba(88,166,255,.2);color:var(--blue);}
.strat-c{background:rgba(227,179,65,.2);color:var(--yellow);}
.strat-d{background:rgba(255,166,87,.2);color:var(--orange);}
.strat-e{background:rgba(163,113,247,.2);color:#a371f7;}
.strat-f{background:rgba(219,39,119,.2);color:#db2777;}
.strat-g{background:rgba(14,165,233,.2);color:#0ea5e9;}
.strat-h{background:rgba(245,158,11,.2);color:#f59e0b;}
.badge-purple{background:rgba(163,113,247,.15);color:#a371f7;}
.badge-pink{background:rgba(219,39,119,.15);color:#db2777;}
.badge-fire{background:rgba(255,100,0,.15);color:#ff6400;font-weight:700;}
.badge-warn{background:rgba(248,81,73,.15);color:var(--red);font-weight:600;}
.strat-info{display:flex;flex-direction:column;gap:2px;flex:1;min-width:0;}
.strat-name{font-size:13px;font-weight:600;}
.strat-desc{font-size:11px;color:var(--text2);}
/* result table */
.result-card{background:var(--bg2);border:1px solid var(--border);border-radius:12px;padding:20px;display:flex;flex-direction:column;gap:10px;}
.result-summary{font-size:13px;color:var(--text2);}
table{width:100%;border-collapse:collapse;font-size:12px;}
thead th{text-align:left;color:var(--text2);font-size:11px;padding:4px 6px;border-bottom:1px solid var(--border);white-space:nowrap;cursor:pointer;user-select:none;}
thead th:hover{color:var(--text);}
tbody td{padding:6px 8px;border-bottom:1px solid var(--border);vertical-align:middle;white-space:nowrap;}
tbody tr:last-child td{border-bottom:none;}
tbody tr:hover{background:rgba(88,166,255,.05);}
.td-code{font-family:'JetBrains Mono',monospace;color:var(--blue);font-weight:600;white-space:nowrap;}
.td-num{font-family:'JetBrains Mono',monospace;white-space:nowrap;}
.code-link{color:var(--blue);text-decoration:none;font-weight:600;}
.code-link:hover{text-decoration:underline;color:#79c0ff;}
.badge{display:inline-block;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:600;white-space:nowrap;}
.badge-green{background:rgba(63,185,80,.15);color:var(--green);}
.badge-red{background:rgba(248,81,73,.15);color:var(--red);}
.badge-yellow{background:rgba(227,179,65,.15);color:var(--yellow);}
.badge-orange{background:rgba(255,166,87,.15);color:var(--orange);}
.badge-blue{background:rgba(88,166,255,.15);color:var(--blue);}
.badge-gray{background:rgba(139,148,158,.15);color:var(--text2);}
.link-btn{background:transparent;border:none;color:var(--blue);font-size:12px;cursor:pointer;padding:2px 6px;border-radius:4px;font-family:'Noto Sans TC',sans-serif;}
.link-btn:hover{background:rgba(88,166,255,.1);}
.empty{color:var(--text2);font-size:13px;text-align:center;padding:30px;}
.sort-arrow{font-size:10px;margin-left:2px;opacity:.5;}
.sort-arrow.asc::after{content:'\\25b2';}
.sort-arrow.desc::after{content:'\\25bc';}
/* monitor */
.monitor-status{font-size:12px;padding:4px 10px;border-radius:99px;background:var(--bg3);color:var(--text2);}
.monitor-alert-row{background:rgba(248,81,73,.08);border-radius:6px;padding:8px 12px;font-size:13px;}
@media(max-width:640px){.strategy-grid{grid-template-columns:1fr;}}
/* ── 產業報告 Modal ── */
.modal-overlay{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:1000;justify-content:center;align-items:center;}
.modal-overlay.active{display:flex;}
.modal-box{background:var(--bg2);border:1px solid var(--border);border-radius:16px;width:92%;max-width:820px;max-height:85vh;overflow-y:auto;padding:32px;position:relative;scrollbar-width:none;-ms-overflow-style:none;}
.modal-box::-webkit-scrollbar{display:none;}
.modal-close{position:absolute;top:12px;right:16px;background:none;border:none;color:var(--text2);font-size:28px;cursor:pointer;line-height:1;padding:4px 8px;border-radius:8px;}
.modal-close:hover{background:var(--bg3);color:var(--text);}
.modal-title{font-size:18px;font-weight:700;color:var(--text);margin-bottom:20px;padding-right:40px;}
.modal-section{margin-bottom:24px;}
.modal-section h3{font-size:15px;font-weight:700;color:var(--blue);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border);}
.modal-section p{font-size:13px;line-height:1.7;color:var(--text);margin-bottom:8px;}
.modal-section ul{padding-left:20px;margin:8px 0;}
.modal-section li{font-size:13px;line-height:1.7;color:var(--text);margin-bottom:4px;}
.modal-section b{color:var(--blue);}
.modal-box table{width:100%;border-collapse:collapse;margin:10px 0;}
.modal-box th{background:var(--bg3);color:var(--text);font-size:12px;font-weight:600;padding:8px 12px;text-align:left;border:1px solid var(--border);}
.modal-box td{font-size:12px;padding:8px 12px;border:1px solid var(--border);color:var(--text);line-height:1.5;}
.modal-loading{text-align:center;padding:60px 20px;font-size:14px;color:var(--text2);}
.modal-loading .spinner{display:inline-block;width:24px;height:24px;border:3px solid var(--border);border-top:3px solid var(--blue);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:12px;}
@keyframes spin{to{transform:rotate(360deg)}}
.modal-footer{text-align:right;font-size:11px;color:var(--text2);margin-top:16px;padding-top:12px;border-top:1px solid var(--border);}
.report-link{cursor:pointer;text-decoration:none;}
.report-link:hover{text-decoration:underline;filter:brightness(1.2);}
</style>
</head>
<body>
__NAV__
<div class="page">

  <!-- 1. 掃描設定 -->
  <div class="card">
    <div class="card-title">全市場策略掃描</div>
    <div style="font-size:12px;color:var(--text2);margin-bottom:4px">自動載入上市櫃清單，過濾流動性（成交額 &gt; 1 億或量 &gt; 1000 張），再跑策略分析</div>

    <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <div style="display:flex;align-items:center;gap:6px;">
        <label style="font-size:13px;color:var(--text2)">執行日期</label>
        <input type="date" id="scanDate" style="background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:6px 10px;color:var(--text);font-size:13px;font-family:'JetBrains Mono',monospace;outline:none;">
      </div>
    </div>

    <div class="card-title" style="margin-top:8px;">策略選擇（可複選）</div>
    <div class="strategy-grid">
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratA" value="A" checked>
        <span class="strat-badge strat-a">A</span>
        <div class="strat-info">
          <span class="strat-name">均線糾結起漲</span>
          <span class="strat-desc">MA5/20 糾結 + 量增突破</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratB" value="B" checked>
        <span class="strat-badge strat-b">B</span>
        <div class="strat-info">
          <span class="strat-name">多頭續強</span>
          <span class="strat-desc">RSI5 黃金交叉 + 強勢量能</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratC" value="C" checked>
        <span class="strat-badge strat-c">C</span>
        <div class="strat-info">
          <span class="strat-name">W底反轉</span>
          <span class="strat-desc">雙底型態 + 頸線突破</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratD" value="D" checked>
        <span class="strat-badge strat-d">D</span>
        <div class="strat-info">
          <span class="strat-name">籌碼同步</span>
          <span class="strat-desc">法人連買 + 技術確認</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratE" value="E" checked>
        <span class="strat-badge strat-e">E</span>
        <div class="strat-info">
          <span class="strat-name">池A：AI回測支撐</span>
          <span class="strat-desc">左側觀察：回測MA20 + RSI&lt;55 + 量縮0.8x</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratF" value="F" checked>
        <span class="strat-badge strat-f">F</span>
        <div class="strat-info">
          <span class="strat-name">池B：AI動能確認</span>
          <span class="strat-desc">右側攻擊：頭頭高底底高 + 四線多排 + 攻擊量</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratG" value="G" checked>
        <span class="strat-badge strat-g">G</span>
        <div class="strat-info">
          <span class="strat-name">朱家泓進場</span>
          <span class="strat-desc">全產業：頭頭高底底高 + 四線多排 + MA20扣抵值</span>
        </div>
      </label>
      <label class="strategy-checkbox">
        <input type="checkbox" id="stratH" value="H" checked>
        <span class="strat-badge strat-h">H</span>
        <div class="strat-info">
          <span class="strat-name">朱家泓最佳⭐</span>
          <span class="strat-desc">ADX(8)趨勢+RSI濾鏡+四線多排（回測年化+37%）</span>
        </div>
      </label>
    </div>
    <div class="ctrl-row" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
      <button class="btn btn-blue" id="runBtn" onclick="runScreener()">開始掃描全市場</button>
      <button class="btn" id="intradayBtn" onclick="runIntradayScan()" style="background:linear-gradient(135deg,#f59e0b,#d97706);color:#fff;border:none;font-weight:700;">📡 盤中即時掃描 H</button>
      <span style="border-left:1px solid var(--border);height:24px;margin:0 4px;"></span>
      <input type="text" id="diagnoseInput" placeholder="股票代號" maxlength="6"
             style="width:80px;padding:6px 10px;border-radius:8px;border:1px solid var(--border);background:var(--bg2);color:var(--text);font-size:13px;font-family:inherit;"
             onkeydown="if(event.key==='Enter')runDiagnoseH()">
      <button class="btn" onclick="runDiagnoseH()" style="background:linear-gradient(135deg,#8b5cf6,#7c3aed);color:#fff;border:none;font-weight:700;">🔍 H 診斷</button>
      <label style="display:flex;align-items:center;gap:4px;cursor:pointer;font-size:13px;color:var(--text2);">
        <input type="checkbox" id="forceRefresh">
        <span>🔄 強制重抓資料</span>
      </label>
      <button class="btn btn-gray btn-sm" id="cancelBtn" onclick="cancelScreener()" style="display:none">取消掃描</button>
      <span style="font-size:12px;color:var(--text2)" id="ctrlNote"></span>
    </div>
    <div id="cacheInfo" style="font-size:11px;color:var(--text2);margin-top:4px;display:none;">
      <span id="cacheInfoText"></span>
    </div>
  </div>

  <!-- 3. 進度 -->
  <div class="progress-card" id="progressCard">
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <div class="progress-text" id="progressText">準備中...</div>
      <div class="progress-text" id="progressPct" style="font-family:'JetBrains Mono',monospace;font-weight:600;"></div>
    </div>
    <div class="progress-bar-bg"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
  </div>

  <!-- 4. 結果 -->
  <div id="resultArea"></div>

  <!-- 5. 盤中量能檢查 -->
  <div class="card">
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
      <div>
        <div class="card-title" style="margin-bottom:2px;">盤中量能檢查</div>
        <div style="font-size:11px;color:var(--text2)">預估全日量 = (當前量 / 開盤分鐘) x 270，預估量 &gt; 昨量 2x 標為警示</div>
      </div>
      <div style="display:flex;gap:8px;align-items:center;">
        <span class="monitor-status" id="monitorStatus"></span>
        <button class="btn btn-gray btn-sm" id="monitorBtn" onclick="runMonitor()">立即檢查</button>
      </div>
    </div>
    <div id="monitorArea"></div>
  </div>

</div>

<script>
var pollTimer=null;
var _allRows=[];
var _sortCol='';
var _sortAsc=true;

// 初始化日期為今天
(function(){
  var d=new Date();
  var s=d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0')+'-'+String(d.getDate()).padStart(2,'0');
  document.getElementById('scanDate').value=s;
})();

function getSelectedStrategies(){
  return ['A','B','C','D','E','F','G','H'].filter(function(s){return document.getElementById('strat'+s).checked;});
}

function setScanning(active){
  document.getElementById('runBtn').disabled=active;
  document.getElementById('runBtn').style.display=active?'none':'inline-block';
  document.getElementById('cancelBtn').style.display=active?'inline-block':'none';
  var pc=document.getElementById('progressCard');
  pc.style.display=active?'flex':'none';
}

function runScreener(){
  var strats=getSelectedStrategies();
  if(!strats.length){alert('請至少選擇一種策略');return;}
  var dateVal=document.getElementById('scanDate').value||null;
  var forceRefresh=document.getElementById('forceRefresh').checked;

  setScanning(true);
  document.getElementById('ctrlNote').textContent='';
  document.getElementById('resultArea').innerHTML='';
  document.getElementById('cacheInfo').style.display='none';
  _allRows=[];
  document.getElementById('progressText').textContent=forceRefresh?'強制重抓資料中...':'載入全市場行情中...';
  document.getElementById('progressPct').textContent='';
  document.getElementById('progressBar').style.width='0%';

  fetch('/api/screener/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({strategies:strats,date:dateVal,force_refresh:forceRefresh})})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){alert(d.error);setScanning(false);return;}
    pollTimer=setInterval(pollStatus,1500);
  })
  .catch(function(){setScanning(false);alert('連線失敗');});
}

function cancelScreener(){
  document.getElementById('cancelBtn').disabled=true;
  document.getElementById('cancelBtn').textContent='取消中...';
  fetch('/api/screener/cancel',{method:'POST'})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){alert(d.error);}
  })
  .catch(function(){});
}

function pollStatus(){
  fetch('/api/screener/status').then(function(r){return r.json();}).then(function(s){
    var pct=0;
    if(s.total>0){
      pct=Math.round(s.progress/s.total*100);
    }else{
      // Phase 2: 從 current 文字解析進度 "載入歷史行情：XXXX（100/711）"
      var m=(s.current||'').match(/(\d+)\/(\d+)\）/);
      if(m){pct=Math.round(parseInt(m[1])/parseInt(m[2])*100);}
    }
    document.getElementById('progressBar').style.width=pct+'%';
    document.getElementById('progressPct').textContent=pct>0?(pct+'%'):'';
    var txt=s.current||'準備中...';
    document.getElementById('progressText').textContent=txt;

    if(s.done){
      clearInterval(pollTimer);
      setScanning(false);
      document.getElementById('cancelBtn').disabled=false;
      document.getElementById('cancelBtn').textContent='取消掃描';
      // 顯示快取資訊
      showCacheInfo(s.cache_info);
      if(s.error){
        document.getElementById('ctrlNote').textContent=s.error==='已取消'?'掃描已取消':('掃描失敗：'+s.error);
        if(s.rows&&s.rows.length){_allRows=s.rows;renderTable(_allRows);document.getElementById('ctrlNote').textContent+='（已掃描 '+s.progress+'/'+s.total+' 檔，符合 '+_allRows.length+' 檔）';}
        return;
      }
      _allRows=s.rows||[];
      document.getElementById('ctrlNote').textContent='掃描 '+s.total+' 檔，符合策略 '+_allRows.length+' 檔';
      renderTable(_allRows);
    }
  });
}

function showCacheInfo(info){
  var el=document.getElementById('cacheInfo');
  var txt=document.getElementById('cacheInfoText');
  if(!info){el.style.display='none';return;}
  el.style.display='block';
  if(info.from_cache){
    txt.innerHTML='📦 <span style="color:#f59e0b;font-weight:600;">使用快取資料</span>（建立時間：'+info.cache_time+'）'
      +'<span style="margin-left:8px;color:var(--text2)">若需最新資料，請勾選「🔄 強制重抓資料」重新掃描</span>';
  }else{
    txt.innerHTML='✅ <span style="color:#10b981;font-weight:600;">即時抓取資料</span>（抓取時間：'+info.cache_time+'）';
  }
}

// ── 結果表格 ──────────────────────────────────────────────
var STRAT_BADGE={'A':'strat-a','B':'strat-b','C':'strat-c','D':'strat-d','E':'strat-e','F':'strat-f','G':'strat-g','H':'strat-h'};
var AI_INDUSTRIES=['半導體業','電腦及週邊設備業','電子零組件業'];
var COLS=[
  {key:'stock_id',       label:'代號'},
  {key:'name',           label:'名稱'},
  {key:'industry',       label:'產業'},
  {key:'strategy_labels',label:'觸發策略'},
  {key:'pool',           label:'追蹤池'},
  {key:'close',          label:'收盤'},
  {key:'change_pct',     label:'漲跌%'},
  {key:'vol_ratio',      label:'量比'},
];
var TOTAL_COLS=8;

function renderTable(rows){
  var wrap=document.getElementById('resultArea');
  if(!rows.length){
    wrap.innerHTML='<div class="result-card"><div class="empty">沒有觸發任何策略的股票</div></div>';
    return;
  }
  var header=COLS.map(function(c){
    var arrow='<span class="sort-arrow'+(_sortCol===c.key?(' '+(_sortAsc?'asc':'desc')):'')+'"></span>';
    return '<th onclick="sortBy(\\''+c.key+'\\')">'+c.label+arrow+'</th>';
  }).join('');
  var body=rows.map(function(r){
    var chgClass=r.change_pct>0?'badge-green':r.change_pct<0?'badge-red':'badge-gray';
    var stratBadges=(r.triggered||[]).map(function(c){
      return '<span class="badge '+(STRAT_BADGE[c]||'badge-gray')+'">'+c+'</span>';
    }).join(' ');
    var ind=r.industry||'';
    var indBadge=ind?(AI_INDUSTRIES.indexOf(ind)>=0?'<span class="badge badge-purple">'+ind+'</span>':'<span style="font-size:11px;color:var(--text2)">'+ind+'</span>'):'<span style="color:var(--text2)">-</span>';
    // 追蹤池 badge
    var poolBadge='-';
    if(r.pool==='A')poolBadge='<span class="badge badge-purple">池A</span>';
    else if(r.pool==='B')poolBadge='<span class="badge badge-pink">池B</span>';
    var pbTag=r.pullback_buy?'<span class="badge badge-fire" style="margin-left:4px;">🎯 回後買上漲點</span>':'';
    var trStyle=r.pullback_buy?' style="background:rgba(255,100,0,.06);"':'';
    // ── 第一列：主要資訊 ──
    var row1='<tr'+trStyle+'>'+
      '<td class="td-code"><a href="/?q='+r.stock_id+'" target="_blank" class="code-link">'+r.stock_id+'</a></td>'+
      '<td>'+r.name+'</td>'+
      '<td>'+indBadge+'</td>'+
      '<td>'+stratBadges+' <span style="font-size:11px;color:var(--text2)">'+r.strategy_labels+'</span>'+pbTag+'</td>'+
      '<td>'+poolBadge+'</td>'+
      '<td class="td-num">'+(r.close!=null?r.close.toFixed(2):'-')+'</td>'+
      '<td><span class="badge '+chgClass+'">'+(r.change_pct>=0?'+':'')+r.change_pct.toFixed(2)+'%</span></td>'+
      '<td class="td-num">'+(r.vol_ratio!=null?r.vol_ratio.toFixed(2)+'x':'-')+'</td>'+
    '</tr>';
    // ── 第二列：補充資訊 ──
    var extras=[];
    // 產業報告連結
    var indEsc=ind.replace(/'/g,"\\'");
    var nameEsc=r.name.replace(/'/g,"\\'");
    extras.push('<a href="#" class="badge badge-blue report-link" onclick="openReport(\\''+r.stock_id+'\\',\\''+nameEsc+'\\',\\''+indEsc+'\\',event)">📊 產業報告</a>');
    extras.push('<a href="#" class="badge strat-h report-link" onclick="runDiagnoseH(\\''+r.stock_id+'\\',event)">🔍 H 診斷</a>');
    var revYoy=r.rev_yoy!=null?(r.rev_yoy>=0?'+':'')+r.rev_yoy.toFixed(1)+'%':null;
    var revClass=r.rev_yoy!=null?(r.rev_yoy>0?'badge-green':r.rev_yoy<0?'badge-red':'badge-gray'):'';
    if(revYoy)extras.push('<span class="badge '+revClass+'">營收 '+revYoy+'</span>');
    var fNet=r.foreign_net!=null?(r.foreign_net>=0?'+':'')+r.foreign_net.toLocaleString()+'張':null;
    var fClass=r.foreign_net!=null?(r.foreign_net>0?'badge-green':r.foreign_net<0?'badge-red':'badge-gray'):'';
    if(fNet)extras.push('<span class="badge '+fClass+'">外資 '+fNet+'</span>');
    var tNet=r.trust_net!=null?(r.trust_net>=0?'+':'')+r.trust_net.toLocaleString()+'張':null;
    var tClass=r.trust_net!=null?(r.trust_net>0?'badge-green':r.trust_net<0?'badge-red':'badge-gray'):'';
    if(tNet)extras.push('<span class="badge '+tClass+'">投信 '+tNet+'</span>');
    // Strategy H 額外指標
    var sd=r.strategy_details||{};
    if(sd.H){
      var h=sd.H;
      extras.push('<span class="badge strat-h" style="font-size:10px;">ADX '+h.adx+' ('+h.trend_strength+')</span>');
      extras.push('<span class="badge strat-h" style="font-size:10px;">RSI '+h.rsi14+'</span>');
      extras.push('<span class="badge badge-green" style="font-size:10px;">📈 進場 '+h.entry_price+'</span>');
      extras.push('<span class="badge badge-red" style="font-size:10px;">🛑 停損 '+h.stop_loss_price+' (-'+h.stop_loss_pct+'%)</span>');
      extras.push('<span class="badge badge-green" style="font-size:10px;">🎯 停利起算 '+h.trail_start_price+' (+'+h.trail_stop_pct+'%後從高點回落15%出場)</span>');
      extras.push('<span class="badge badge-gray" style="font-size:10px;">⏱ 最少持有'+h.min_hold_days+'天 | 風報比 1:'+h.rr_ratio+'</span>');
    }
    // 買點診斷：僅顯示不通過的步驟
    var steps=r.pullback_steps||[];
    var failSteps=steps.filter(function(s){return !s.ok;});
    if(failSteps.length>0&&failSteps.length<4){
      failSteps.forEach(function(s){
        extras.push('<span class="badge badge-red" style="font-size:10px;">✗ '+s.name+' '+s.desc+'</span>');
      });
    }else if(steps.length&&failSteps.length===0){
      extras.push('<span class="badge badge-fire" style="font-size:10px;">✓ 四步全通過</span>');
    }
    var row2='';
    if(extras.length){
      row2='<tr'+trStyle+'>'+
        '<td colspan="3" style="border-top:none;padding:0;"></td>'+
        '<td colspan="5" style="border-top:none;padding-top:0;padding-bottom:10px;">'+
        '<div style="display:flex;gap:6px;flex-wrap:wrap;">'+extras.join('')+'</div>'+
      '</td></tr>';
    }
    return row1+row2;
  }).join('');
  wrap.innerHTML='<div class="result-card">'+
    '<div class="result-summary">符合策略：<strong>'+rows.length+'</strong> 檔（點欄位標題排序）</div>'+
    '<div style="overflow-x:auto"><table><thead><tr>'+header+'</tr></thead><tbody>'+body+'</tbody></table></div>'+
  '</div>';
}

function sortBy(col){
  if(_sortCol===col){_sortAsc=!_sortAsc;}else{_sortCol=col;_sortAsc=true;}
  var sorted=_allRows.slice().sort(function(a,b){
    var av=a[col],bv=b[col];
    if(av==null)return 1;if(bv==null)return -1;
    if(typeof av==='string')return _sortAsc?av.localeCompare(bv):bv.localeCompare(av);
    return _sortAsc?av-bv:bv-av;
  });
  renderTable(sorted);
}

// ── 盤中即時掃描 H ──────────────────────────────────────────
var intradayPollTimer=null;

function runIntradayScan(){
  document.getElementById('intradayBtn').disabled=true;
  document.getElementById('ctrlNote').textContent='📡 盤中即時掃描啟動...';
  fetch('/api/screener/intraday/run',{method:'POST'})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){alert(d.error);document.getElementById('intradayBtn').disabled=false;document.getElementById('ctrlNote').textContent='';return;}
    intradayPollTimer=setInterval(pollIntraday,1500);
  })
  .catch(function(){document.getElementById('intradayBtn').disabled=false;alert('連線失敗');});
}

function pollIntraday(){
  fetch('/api/screener/intraday/status').then(function(r){return r.json();}).then(function(s){
    document.getElementById('ctrlNote').textContent='📡 '+s.current+' ('+s.progress+'/'+s.total+')';
    if(!s.done)return;
    clearInterval(intradayPollTimer);
    document.getElementById('intradayBtn').disabled=false;
    if(s.error){document.getElementById('ctrlNote').textContent='❌ '+s.error;return;}
    document.getElementById('ctrlNote').textContent='✅ 盤中掃描完成（即時報價）';
    if(s.rows&&s.rows.length>0){
      _allRows=s.rows;
      renderTable(s.rows);
      document.getElementById('statusText').textContent='📡 盤中即時 H 策略命中：'+s.rows.length+' 檔';
    }else{
      document.getElementById('statusText').textContent='📡 盤中掃描完成：目前無 H 策略訊號';
      var wrap=document.getElementById('resultWrap');
      wrap.innerHTML='<div class="result-card"><div class="result-summary">📡 盤中即時掃描完成 — 目前無符合 H 策略的進場訊號</div></div>';
    }
  });
}

// ── H 策略逐條診斷 ──────────────────────────────────────────

function runDiagnoseH(stockId, evt){
  if(evt){evt.preventDefault();evt.stopPropagation();}
  var sid=stockId||document.getElementById('diagnoseInput').value.trim();
  if(!sid){alert('請輸入股票代號');return;}

  var modal=document.getElementById('diagnoseModal');
  var content=document.getElementById('diagnoseContent');
  modal.classList.add('active');
  document.body.style.overflow='hidden';

  content.innerHTML='<div class="modal-loading"><div class="spinner"></div><br>正在診斷 <b>'+sid+'</b> H 策略條件...</div>';

  fetch('/api/screener/diagnose-h',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_id:sid})
  })
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){
      content.innerHTML='<div class="modal-loading" style="color:var(--red);">'+d.error+'</div>';
      return;
    }
    _renderDiagnose(d);
  })
  .catch(function(e){
    content.innerHTML='<div class="modal-loading" style="color:var(--red);">診斷失敗：'+e+'</div>';
  });
}

function _renderDiagnose(d){
  var content=document.getElementById('diagnoseContent');
  var allPassed=d.passed;
  var cnt=d.passed_count;
  var total=d.total_checks;
  var s=d.summary||{};

  var html='<div class="modal-title" style="display:flex;align-items:center;gap:10px;">';
  html+='<span>'+d.stock_id+' '+(d.name||'')+'</span>';
  if(allPassed){
    html+='<span style="background:rgba(52,211,153,.2);color:#34d399;padding:3px 10px;border-radius:6px;font-size:13px;font-weight:700;">ALL PASS</span>';
  }else{
    html+='<span style="background:rgba(248,81,73,.15);color:#f85149;padding:3px 10px;border-radius:6px;font-size:13px;font-weight:700;">'+cnt+' / '+total+' 通過</span>';
  }
  html+='</div>';

  // 摘要
  html+='<div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;">';
  var tags=[
    {l:'收盤',v:s.close},
    {l:'ADX',v:s.adx},
    {l:'RSI',v:s.rsi14},
    {l:'MA5',v:s.ma5},
    {l:'MA60',v:s.ma60}
  ];
  tags.forEach(function(t){
    html+='<span style="background:var(--bg3);padding:4px 10px;border-radius:6px;font-size:12px;">'+t.l+' <b>'+t.v+'</b></span>';
  });
  html+='</div>';

  // 逐條
  html+='<div style="display:flex;flex-direction:column;gap:8px;">';
  var checks=d.checks||[];
  for(var i=0;i<checks.length;i++){
    var c=checks[i];
    var bg=c.passed?'rgba(52,211,153,.08)':'rgba(248,81,73,.08)';
    var border=c.passed?'rgba(52,211,153,.3)':'rgba(248,81,73,.3)';
    var icon=c.passed?'✅':'❌';

    html+='<div style="background:'+bg+';border:1px solid '+border+';border-radius:10px;padding:12px 16px;">';
    html+='<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">';
    html+='<div style="display:flex;align-items:center;gap:8px;">';
    html+='<span style="font-size:16px;">'+icon+'</span>';
    html+='<span style="font-weight:700;font-size:14px;color:var(--text);">'+(i+1)+'. '+c.name+'</span>';
    html+='</div>';
    html+='<span style="font-size:12px;color:var(--text2);background:var(--bg2);padding:2px 8px;border-radius:4px;">'+c.detail+'</span>';
    html+='</div>';
    html+='<div style="margin-top:6px;font-size:12px;color:var(--text2);display:flex;justify-content:space-between;">';
    html+='<span>實際值：<b style="color:var(--text);">'+c.value+'</b></span>';
    html+='<span>門檻：'+c.threshold+'</span>';
    html+='</div>';
    html+='</div>';
  }
  html+='</div>';

  content.innerHTML=html;
}

function closeDiagnoseModal(evt){
  if(!evt||evt.target.classList.contains('modal-overlay')){
    document.getElementById('diagnoseModal').classList.remove('active');
    document.body.style.overflow='';
  }
}

// ESC 也能關閉診斷 Modal
(function(){
  var _origKeydown=document.onkeydown;
  document.addEventListener('keydown',function(e){
    if(e.key==='Escape'){closeDiagnoseModal();}
  });
})();

// ── 盤中量能檢查 ──────────────────────────────────────────
var monitorPollTimer=null;

function runMonitor(){
  var ids=_allRows.map(function(r){return r.stock_id;});
  if(!ids.length){alert('請先執行策略掃描，取得結果後再檢查量能');return;}
  document.getElementById('monitorBtn').disabled=true;
  document.getElementById('monitorStatus').textContent='檢查中...';
  document.getElementById('monitorArea').innerHTML='';

  fetch('/api/screener/monitor/run',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_ids:ids})})
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){alert(d.error);document.getElementById('monitorBtn').disabled=false;document.getElementById('monitorStatus').textContent='';return;}
    monitorPollTimer=setInterval(pollMonitor,1500);
  })
  .catch(function(){document.getElementById('monitorBtn').disabled=false;alert('連線失敗');});
}

function pollMonitor(){
  fetch('/api/screener/monitor/status').then(function(r){return r.json();}).then(function(s){
    if(!s.done&&s.running)return;
    clearInterval(monitorPollTimer);
    document.getElementById('monitorBtn').disabled=false;
    document.getElementById('monitorStatus').textContent=s.last_check?('最後檢查：'+s.last_check):'';
    if(s.error){alert('檢查失敗：'+s.error);return;}
    renderMonitorAlerts(s.alerts||[]);
  });
}

function renderMonitorAlerts(alerts){
  var wrap=document.getElementById('monitorArea');
  if(!alerts.length){wrap.innerHTML='<div style="font-size:13px;color:var(--text2);padding:8px 0">無資料或非盤中時段</div>';return;}
  var html=alerts.map(function(a){
    var cls=a.alert?'monitor-alert-row':'';
    var icon=a.alert?'\\u26a0\\ufe0f ':'';
    return '<div class="'+cls+'" style="padding:6px 12px;border-bottom:1px solid var(--border);font-size:13px;">'+
      icon+'<span class="td-code">'+a.stock_id+'</span>'+
      ' <span class="td-num">'+a.price.toFixed(2)+'</span>'+
      ' 預估量比：<strong>'+(a.vol_ratio!=null?a.vol_ratio.toFixed(2)+'x':'-')+'</strong>'+
      (a.alert?' <span class="badge badge-red">異常放量</span>':'')+
    '</div>';
  }).join('');
  wrap.innerHTML=html;
}

// ── 頁面載入時自動顯示上次結果 ──────────────────────────────
(function(){
  fetch('/api/screener/status').then(function(r){return r.json();}).then(function(s){
    if(s.running){
      setScanning(true);
      pollTimer=setInterval(pollStatus,2000);
      return;
    }
    if(s.done && s.rows && s.rows.length){
      _allRows=s.rows;
      renderTable(_allRows);
      document.getElementById('ctrlNote').textContent='上次掃描：'+s.total+' 檔，符合策略 '+_allRows.length+' 檔';
    }
  }).catch(function(){});
})();

// ── 產業報告 Modal ──────────────────────────────────────────
var _reportCache={};

function openReport(sid, name, industry, evt){
  if(evt)evt.preventDefault();
  var modal=document.getElementById('reportModal');
  var content=document.getElementById('reportContent');
  modal.classList.add('active');
  document.body.style.overflow='hidden';

  // 前端快取命中
  if(_reportCache[sid]){
    _renderReport(_reportCache[sid]);
    return;
  }

  content.innerHTML='<div class="modal-loading"><div class="spinner"></div><br>AI 正在生成 <b>'+name+'</b> 產業分析報告...<br><span style="font-size:12px;color:var(--text2);">首次生成約需 10-15 秒，之後走快取</span></div>';

  fetch('/api/industry-report',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({stock_id:sid, name:name, industry:industry})
  })
  .then(function(r){return r.json();})
  .then(function(d){
    if(d.error){
      content.innerHTML='<div class="modal-loading" style="color:var(--red);">❌ '+d.error+'</div>';
      return;
    }
    _reportCache[sid]=d;
    _renderReport(d);
  })
  .catch(function(e){
    content.innerHTML='<div class="modal-loading" style="color:var(--red);">❌ 報告生成失敗：'+e+'</div>';
  });
}

function _renderReport(d){
  var content=document.getElementById('reportContent');
  var s=d.sections||{};
  var html='<div class="modal-title">'+d.ticker+' '+d.name+'：產業地位與展望分析報告</div>';
  if(d.error_hint){
    html+='<div style="background:rgba(248,81,73,.1);border:1px solid rgba(248,81,73,.3);border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:12px;color:var(--red);">⚠️ '+d.error_hint+'</div>';
  }
  html+='<div class="modal-section"><h3>1. 產業定位與核心業務</h3>'+(s.positioning||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>2. 產業展望與利多題材</h3>'+(s.growth||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>3. 同類型競爭對手</h3>'+(s.peers||'<p>無資料</p>')+'</div>';
  html+='<div class="modal-section"><h3>4. 法說會資訊</h3>'+(s.conference||'<p>無資料</p>')+'</div>';
  if(d.generated_at){
    html+='<div class="modal-footer">報告生成：'+d.generated_at+(d.cached?' ✅ 快取命中（7 天有效）':' 🤖 Claude AI 即時生成')+'</div>';
  }
  content.innerHTML=html;
}

function closeReportModal(evt){
  if(!evt||evt.target.classList.contains('modal-overlay')){
    document.getElementById('reportModal').classList.remove('active');
    document.body.style.overflow='';
  }
}

// ESC 關閉 Modal
document.addEventListener('keydown',function(e){
  if(e.key==='Escape'){closeReportModal();}
});
</script>
<!-- 產業報告 Modal -->
<div class="modal-overlay" id="reportModal" onclick="closeReportModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeReportModal()">&times;</button>
    <div id="reportContent">
      <div class="modal-loading"><div class="spinner"></div><br>AI 產業報告生成中...</div>
    </div>
  </div>
</div>

<div class="modal-overlay" id="diagnoseModal" onclick="closeDiagnoseModal(event)">
  <div class="modal-box" style="max-width:620px;" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeDiagnoseModal()">&times;</button>
    <div id="diagnoseContent">
      <div class="modal-loading"><div class="spinner"></div><br>載入中...</div>
    </div>
  </div>
</div>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════
# 路由：頁面
# ═══════════════════════════════════════════════════════════

def _render(template: str, nav_active: str) -> str:
    """用字串取代而非 .format()，避免 JS 中的 {} 被 Python 誤解析"""
    return (template
            .replace("__NAV_STYLE__", _NAV_STYLE)
            .replace("__NAV__", _nav(nav_active)))


@app.route("/")
def page_query():
    return _render(QUERY_PAGE, "/"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/portfolio")
def page_portfolio():
    return _render(PORTFOLIO_PAGE, "/portfolio"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/batch")
def page_batch():
    return _render(BATCH_PAGE, "/batch"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/screener")
def page_screener():
    return _render(SCREENER_PAGE, "/screener"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/chu-review")
def page_chu_review():
    return _render(CHU_REVIEW_PAGE, "/chu-review"), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/h-diagnose")
def page_h_diagnose():
    return _render(H_DIAGNOSE_PAGE, "/h-diagnose"), 200, {"Content-Type": "text/html; charset=utf-8"}


# ═══════════════════════════════════════════════════════════
# 路由：Portfolio API
# ═══════════════════════════════════════════════════════════

@app.route("/api/portfolio", methods=["GET"])
def api_portfolio_get():
    return jsonify({"portfolio": ps.get_all()})


@app.route("/api/portfolio", methods=["POST"])
def api_portfolio_upsert():
    data = request.json or {}
    try:
        ps.upsert(
            stock_id   = data["stock_id"],
            name       = data.get("name", data["stock_id"]),
            shares     = float(data["shares"]),
            cost_price = float(data["cost_price"]),
        )
        return jsonify({"ok": True})
    except (KeyError, ValueError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/portfolio/<stock_id>", methods=["DELETE"])
def api_portfolio_delete(stock_id):
    ps.remove(stock_id)
    return jsonify({"ok": True})


@app.route("/api/portfolio/import", methods=["POST"])
def api_portfolio_import():
    csv_text = (request.json or {}).get("csv", "")
    count, err = ps.import_csv_text(csv_text)
    if err:
        return jsonify({"ok": False, "error": err}), 400
    return jsonify({"ok": True, "count": count})


@app.route("/api/portfolio/export", methods=["GET"])
def api_portfolio_export():
    csv_text = ps.export_csv_text()
    filename = f"portfolio_{datetime.today().strftime('%Y%m%d')}.csv"
    return Response(
        csv_text,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ═══════════════════════════════════════════════════════════
# 路由：單支查詢 API
# ═══════════════════════════════════════════════════════════

@app.route("/api/query", methods=["POST"])
def api_query():
    data     = request.json or {}
    stock_id = data.get("stock_id", "").strip()
    force    = data.get("force", False)
    today    = datetime.today().strftime("%Y-%m-%d")

    if not stock_id:
        return jsonify({"error": "請輸入股票代號"}), 400

    cache_key = f"{stock_id.zfill(4)}_{today}"
    if not force and cache_key in _query_cache:
        cached = _query_cache[cache_key]
        return jsonify({
            "html":     cached["card_html"],
            "name":     cached["name"],
            "action":   cached["analysis"].get("action", "觀望"),
            "analysis": cached["analysis"],
            "signals":  cached["signals"],
            "cached":   True,
        })

    try:
        result, err = _query_stock(stock_id)
        if err:
            return jsonify({"error": err}), 404
        _query_cache[cache_key] = result
        return jsonify({
            "html":     result["card_html"],
            "name":     result["name"],
            "action":   result["analysis"].get("action", "觀望"),
            "analysis": result["analysis"],
            "signals":  result["signals"],
            "cached":   False,
        })
    except Exception as e:
        return jsonify({"error": f"查詢失敗：{e}"}), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    today = datetime.today().strftime("%Y-%m-%d")
    items = []
    for key, val in _query_cache.items():
        if key.endswith(f"_{today}"):
            items.append({
                "stock_id":  val["stock_id"],
                "name":      val["name"],
                "action":    val["analysis"].get("action", "觀望"),
                "timestamp": today,
            })
    return jsonify({"history": items})


# ═══════════════════════════════════════════════════════════
# 路由：批次分析 API
# ═══════════════════════════════════════════════════════════

@app.route("/api/batch/run", methods=["POST"])
def api_batch_run():
    global _batch_status
    if _batch_status["running"]:
        return jsonify({"error": "分析進行中，請稍候"}), 409

    force        = (request.json or {}).get("force", False)
    portfolio_df = ps.to_dataframe()
    if portfolio_df.empty:
        return jsonify({"error": "持股清單為空，請先至「持股管理」新增或匯入 CSV"}), 400

    today = datetime.today().strftime("%Y-%m-%d")
    _batch_status.update({
        "running":     True,
        "progress":    0,
        "total":       len(portfolio_df),
        "current":     "",
        "done":        False,
        "error":       None,
        "result":      [],
        "report_html": "",
    })

    t = threading.Thread(
        target=_run_batch,
        args=(portfolio_df, force, today),
        daemon=True,
    )
    t.start()
    return jsonify({"status": "started", "total": len(portfolio_df)})


@app.route("/api/batch/status", methods=["GET"])
def api_batch_status():
    return jsonify({
        "running":     _batch_status["running"],
        "progress":    _batch_status["progress"],
        "total":       _batch_status["total"],
        "current":     _batch_status["current"],
        "done":        _batch_status["done"],
        "error":       _batch_status["error"],
        "report_html": _batch_status["report_html"],
    })


# ═══════════════════════════════════════════════════════════
# 路由：盤中即時掃描 API
# ═══════════════════════════════════════════════════════════

@app.route("/api/screener/intraday/run", methods=["POST"])
def api_intraday_run():
    global _intraday_status
    if _intraday_status.get("running"):
        return jsonify({"error": "盤中掃描進行中"}), 409

    _intraday_status = {
        "running": True, "done": False, "progress": 0,
        "total": 0, "current": "啟動中...", "error": None, "rows": [],
    }
    t = threading.Thread(target=_run_intraday_scan, daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/screener/intraday/status", methods=["GET"])
def api_intraday_status():
    return jsonify({
        "running":  _intraday_status["running"],
        "done":     _intraday_status["done"],
        "progress": _intraday_status["progress"],
        "total":    _intraday_status["total"],
        "current":  _intraday_status["current"],
        "error":    _intraday_status["error"],
        "rows":     _intraday_status["rows"],
    })


# ── H 策略逐條診斷 ──

@app.route("/api/screener/diagnose-h", methods=["POST"])
def api_diagnose_h():
    """H 策略逐條診斷 — 輸入股票代號，回傳每個條件的通過/失敗。"""
    from data_fetcher import fetch_stock_price_public
    from screener import compute_screener_indicators
    from strategies.master_chu import diagnose_h_strategy

    data = request.get_json() or {}
    stock_id = data.get("stock_id", "").strip()

    if not stock_id:
        return jsonify({"error": "請輸入股票代號"}), 400

    try:
        # 嘗試 twse，失敗再試 tpex
        df = fetch_stock_price_public(stock_id, exchange="twse", months=6)
        if df.empty:
            df = fetch_stock_price_public(stock_id, exchange="tpex", months=6)
        if df.empty:
            return jsonify({"error": f"無法取得 {stock_id} 的歷史資料"}), 404

        enriched = compute_screener_indicators(df)
        result = diagnose_h_strategy(enriched)
        result["stock_id"] = stock_id

        # 嘗試取得股票名稱
        if "name" in df.columns:
            result["name"] = str(df.iloc[-1]["name"])
        else:
            result["name"] = stock_id

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"診斷失敗：{e}"}), 500


# ═══════════════════════════════════════════════════════════
# 路由：選股 API
# ═══════════════════════════════════════════════════════════

@app.route("/api/screener/run", methods=["POST"])
def api_screener_run():
    global _screener_status
    if _screener_status["running"]:
        return jsonify({"error": "掃描進行中，請稍候"}), 409

    data          = request.json or {}
    strategies    = data.get("strategies", ["A", "B", "C", "D", "E", "F"])
    target_date   = data.get("date", None)  # 可選執行日期
    force_refresh = data.get("force_refresh", False)

    if not strategies:
        return jsonify({"error": "請至少選擇一種策略"}), 400

    _screener_status.update({
        "running":    True,
        "progress":   0,
        "total":      0,
        "current":    "準備中...",
        "done":       False,
        "error":      None,
        "rows":       [],
        "strategies": strategies,
        "cancel":     False,
        "cache_info": None,
    })

    t = threading.Thread(
        target=_run_screener,
        args=(strategies, target_date, force_refresh),
        daemon=True,
    )
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/screener/cancel", methods=["POST"])
def api_screener_cancel():
    if not _screener_status["running"]:
        return jsonify({"error": "目前沒有執行中的掃描"}), 400
    _screener_status["cancel"] = True
    return jsonify({"status": "cancelling"})


@app.route("/api/screener/status", methods=["GET"])
def api_screener_status():
    return jsonify(_sanitize_for_json({
        "running":    _screener_status["running"],
        "progress":   _screener_status["progress"],
        "total":      _screener_status["total"],
        "current":    _screener_status["current"],
        "done":       _screener_status["done"],
        "error":      _screener_status["error"],
        "rows":       _screener_status["rows"],
        "strategies": _screener_status["strategies"],
        "cache_info": _screener_status.get("cache_info"),
    }))


@app.route("/api/screener/watchlist", methods=["GET"])
def api_screener_watchlist():
    """回傳 watchlist.json 追蹤清單（供除錯/查看）。"""
    try:
        from watchlist import load_watchlist
        return jsonify(load_watchlist())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/screener/monitor/run", methods=["POST"])
def api_monitor_run():
    global _monitor_status
    if _monitor_status.get("running"):
        return jsonify({"error": "盤中檢查進行中，請稍候"}), 409

    stock_ids = (request.json or {}).get("stock_ids", [])
    if not stock_ids:
        return jsonify({"error": "請提供監控股票清單"}), 400

    t = threading.Thread(target=_run_monitor, args=(stock_ids,), daemon=True)
    t.start()
    return jsonify({"status": "started", "stock_count": len(stock_ids)})


@app.route("/api/screener/monitor/status", methods=["GET"])
def api_monitor_status():
    return jsonify({
        "running":    _monitor_status["running"],
        "done":       _monitor_status["done"],
        "error":      _monitor_status["error"],
        "last_check": _monitor_status["last_check"],
        "alerts":     _monitor_status["alerts"],
        "stock_ids":  _monitor_status["stock_ids"],
    })


# ═══════════════════════════════════════════════════════════
# 路由：AI 產業報告
# ═══════════════════════════════════════════════════════════

@app.route("/api/industry-report", methods=["POST"])
def api_industry_report():
    """AI 產業分析報告（Gemini）"""
    from industry_analyst import get_industry_report

    data = request.get_json() or {}
    stock_id = data.get("stock_id", "")
    name = data.get("name", stock_id)
    industry = data.get("industry", "")

    if not stock_id:
        return jsonify({"error": "缺少 stock_id"}), 400

    try:
        report = get_industry_report(stock_id, name, industry)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": f"產業報告生成失敗：{e}"}), 500


# ═══════════════════════════════════════════════════════════
# 路由：Telegram API
# ═══════════════════════════════════════════════════════════

@app.route("/api/telegram/single", methods=["POST"])
def api_telegram_single():
    from notifier import send_telegram

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "未設定 Telegram Bot Token 或 Chat ID"}), 400

    data     = request.json or {}
    stock_id = data.get("stock_id", "")
    name     = data.get("name", stock_id)
    analysis = data.get("analysis", {})
    signals  = data.get("signals", {})

    close      = signals.get("close", 0)
    change     = signals.get("change", 0)
    change_pct = signals.get("change_pct", 0)
    rsi        = signals.get("rsi", "-")
    macd_hist  = signals.get("macd_hist", 0) or 0
    action     = analysis.get("action", "觀望")

    action_emoji = {"持有": "🔵", "加碼": "🟢", "減碼": "🟠", "觀望": "⚪", "停損": "🔴"}.get(action, "⚪")
    arrow        = "📈" if change > 0 else "📉"

    stop_price = analysis.get("stop_loss_price", "—")
    rev_bull   = analysis.get("reversal_bull", "—")
    rev_bear   = analysis.get("reversal_bear", "—")

    msg = (
        f"🔔 {stock_id} {name}\n"
        f"{arrow} {close}（{change:+.1f}, {change_pct:+.2f}%）\n"
        f"RSI {rsi} ｜ MACD {macd_hist:+.3f}\n\n"
        f"🎯 操作摘要\n"
        f"━━━━━━━━━━━━━\n"
        f"{action_emoji} {action}：{analysis.get('action_reason', '')}\n\n"
        f"🛑 停損：{stop_price} 元\n"
        f"🔼 多方反轉：{rev_bull}\n"
        f"🔽 空方確認：{rev_bear}\n\n"
        f"⚠️ {analysis.get('risk', '—')}"
    )

    ok = send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, parse_mode="")
    if ok:
        return jsonify({"success": True})
    return jsonify({"error": "Telegram 傳送失敗"}), 500


@app.route("/api/telegram/batch", methods=["POST"])
def api_telegram_batch():
    from notifier import send_telegram, build_summary_message

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({"error": "未設定 Telegram Bot Token 或 Chat ID"}), 400

    results = _batch_status.get("result", [])
    if not results:
        return jsonify({"error": "尚無批次分析結果"}), 400

    report_date = datetime.today().strftime("%Y/%m/%d")
    msg = build_summary_message(results, report_date)
    ok  = send_telegram(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
    if ok:
        return jsonify({"success": True})
    return jsonify({"error": "Telegram 傳送失敗"}), 500


# ═══════════════════════════════════════════════════════════
# 路由：朱家泓覆盤 API（背景線程 + polling）
# ═══════════════════════════════════════════════════════════

_chu_review_status = {
    "running": False, "done": False, "error": None,
    "progress": 0, "total": 0, "current": "",
    "result": None,
}


def _run_chu_review_bg(stock_ids):
    """背景線程執行覆盤，避免 gunicorn timeout。"""
    global _chu_review_status
    try:
        _chu_review_status["running"] = True
        _chu_review_status["done"] = False
        _chu_review_status["error"] = None
        _chu_review_status["result"] = None

        from strategies import discover_strategies, get_strategy
        from screener import compute_screener_indicators
        from data_fetcher import fetch_price, fetch_realtime_quote, fetch_industry_map
        from datetime import date as _date

        discover_strategies()
        chu_info = get_strategy("G")
        if chu_info is None or chu_info.review_func is None:
            _chu_review_status["error"] = "朱家泓覆盤策略未註冊"
            return

        _chu_review_status["current"] = "載入產業分類..."
        try:
            ind_map = fetch_industry_map()
        except Exception:
            ind_map = {}

        today_str = _date.today().strftime("%Y-%m-%d")
        reviews = []
        summary = {"total": 0, "healthy": 0, "reduce": 0, "alert": 0, "take_profit": 0, "buy_point": 0}
        is_intraday = False
        _chu_review_status["total"] = len(stock_ids)

        for i, sid in enumerate(stock_ids):
            _chu_review_status["progress"] = i + 1
            _chu_review_status["current"] = f"覆盤 {sid}（{i+1}/{len(stock_ids)}）"

            try:
                price_df = fetch_price(sid)
                if price_df.empty or len(price_df) < 20:
                    reviews.append({
                        "stock_id": sid, "name": sid, "close": None, "change_pct": 0,
                        "review": {"status": "healthy", "signals": [], "ma_status": {}, "k_bar": {}, "summary": "無足夠資料"},
                    })
                    summary["total"] += 1
                    summary["healthy"] += 1
                    continue

                rt = fetch_realtime_quote(sid)
                rt_time = None
                if rt and rt["price"]:
                    rt_date_str = rt.get("date", "")
                    rt_time = rt.get("time", "")
                    if rt_date_str:
                        rt_date = pd.Timestamp(f"{rt_date_str[:4]}-{rt_date_str[4:6]}-{rt_date_str[6:8]}")
                        last_hist_date = price_df["date"].iloc[-1]
                        if rt_date > last_hist_date:
                            new_row = pd.DataFrame([{
                                "date": rt_date,
                                "open": rt.get("open") or rt["price"],
                                "high": rt.get("high") or rt["price"],
                                "low": rt.get("low") or rt["price"],
                                "close": rt["price"],
                                "volume": rt.get("volume", 0),
                            }])
                            price_df = pd.concat([price_df, new_row], ignore_index=True)
                            is_intraday = True
                        elif rt_date == last_hist_date:
                            price_df.loc[price_df.index[-1], "close"] = rt["price"]
                            if rt.get("high"):
                                price_df.loc[price_df.index[-1], "high"] = rt["high"]
                            if rt.get("low"):
                                price_df.loc[price_df.index[-1], "low"] = rt["low"]
                            if rt.get("volume"):
                                price_df.loc[price_df.index[-1], "volume"] = rt["volume"]
                            is_intraday = True

                enriched = compute_screener_indicators(price_df)
                last = enriched.iloc[-1]
                prev = enriched.iloc[-2] if len(enriched) >= 2 else last
                close = float(last["close"])
                change_pct = round((last["close"] - prev["close"]) / prev["close"] * 100, 2) if prev["close"] else 0

                name = rt.get("name", sid) if rt else sid
                if name == sid and "stock_name" in price_df.columns:
                    name = str(price_df["stock_name"].iloc[-1]) if pd.notna(price_df["stock_name"].iloc[-1]) else sid
                holding = ps.get_by_id(sid)
                if holding:
                    name = holding.get("name", name)

                review = chu_info.review_func(enriched)
                review_item = {
                    "stock_id": sid, "name": name, "industry": ind_map.get(sid, ""),
                    "close": close, "change_pct": change_pct, "review": review,
                }
                if rt_time:
                    review_item["rt_time"] = rt_time
                reviews.append(review_item)

                summary["total"] += 1
                st = review.get("status", "healthy")
                if st in summary:
                    summary[st] += 1

            except Exception as e:
                print(f"朱家泓覆盤 {sid} 失敗：{e}")
                reviews.append({
                    "stock_id": sid, "name": sid, "close": None, "change_pct": 0,
                    "review": {"status": "healthy", "signals": [], "ma_status": {}, "k_bar": {}, "summary": f"覆盤失敗：{e}"},
                })
                summary["total"] += 1
                summary["healthy"] += 1

        _chu_review_status["result"] = _sanitize_for_json({
            "date": today_str, "reviews": reviews, "summary": summary, "is_intraday": is_intraday,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        _chu_review_status["error"] = str(e)
    finally:
        _chu_review_status["done"] = True
        _chu_review_status["running"] = False


@app.route("/api/chu-review/run", methods=["POST"])
def api_chu_review_run():
    global _chu_review_status
    if _chu_review_status.get("running"):
        return jsonify({"started": True, "msg": "已在執行中"})

    data = request.json or {}
    stock_ids = data.get("stock_ids", [])
    if not stock_ids:
        stock_ids = [p["stock_id"] for p in ps.get_all()]
    if not stock_ids:
        return jsonify({"error": "請提供股票代號或先建立持股"}), 400

    _chu_review_status = {
        "running": False, "done": False, "error": None,
        "progress": 0, "total": len(stock_ids), "current": "啟動中...",
        "result": None,
    }
    t = threading.Thread(target=_run_chu_review_bg, args=(stock_ids,), daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/api/chu-review/status", methods=["GET"])
def api_chu_review_status():
    return jsonify({
        "running": _chu_review_status["running"],
        "done": _chu_review_status["done"],
        "progress": _chu_review_status["progress"],
        "total": _chu_review_status["total"],
        "current": _chu_review_status["current"],
        "error": _chu_review_status["error"],
        "result": _chu_review_status["result"],
    })


# ═══════════════════════════════════════════════════════════
# 路由：H策略即時診斷 API（背景線程 + polling）
# ═══════════════════════════════════════════════════════════

_h_diagnose_status = {
    "running": False, "done": False, "error": None,
    "progress": 0, "total": 0, "current": "",
    "result": None,
}


def _run_h_diagnose_bg(stock_ids):
    """背景線程執行 H 策略診斷，避免 gunicorn timeout。"""
    global _h_diagnose_status
    try:
        _h_diagnose_status["running"] = True
        _h_diagnose_status["done"] = False
        _h_diagnose_status["error"] = None
        _h_diagnose_status["result"] = None

        from screener import compute_screener_indicators
        from data_fetcher import fetch_price, fetch_realtime_quote
        from strategies.master_chu import diagnose_h_strategy
        from datetime import date as _date

        today_str = _date.today().strftime("%Y-%m-%d")
        results = []
        is_intraday = False
        _h_diagnose_status["total"] = len(stock_ids)

        for i, sid in enumerate(stock_ids):
            _h_diagnose_status["progress"] = i + 1
            _h_diagnose_status["current"] = f"診斷 {sid}（{i+1}/{len(stock_ids)}）"

            try:
                price_df = fetch_price(sid)
                if price_df.empty or len(price_df) < 65:
                    results.append({
                        "stock_id": sid, "name": sid,
                        "close": None,
                        "diagnose": {
                            "passed": False, "passed_count": 0,
                            "total_checks": 9, "checks": [], "summary": {},
                            "error": "歷史資料不足（需至少 65 天）",
                        },
                    })
                    continue

                # 即時報價：合併到歷史資料的最新一筆
                rt = fetch_realtime_quote(sid)
                rt_time = None
                if rt and rt["price"]:
                    rt_date_str = rt.get("date", "")
                    rt_time = rt.get("time", "")
                    if rt_date_str:
                        rt_date = pd.Timestamp(
                            f"{rt_date_str[:4]}-{rt_date_str[4:6]}-{rt_date_str[6:8]}"
                        )
                        last_hist_date = price_df["date"].iloc[-1]
                        if rt_date > last_hist_date:
                            new_row = pd.DataFrame([{
                                "date": rt_date,
                                "open": rt.get("open") or rt["price"],
                                "high": rt.get("high") or rt["price"],
                                "low": rt.get("low") or rt["price"],
                                "close": rt["price"],
                                "volume": rt.get("volume", 0),
                            }])
                            price_df = pd.concat([price_df, new_row], ignore_index=True)
                            is_intraday = True
                        elif rt_date == last_hist_date:
                            price_df.loc[price_df.index[-1], "close"] = rt["price"]
                            if rt.get("high"):
                                price_df.loc[price_df.index[-1], "high"] = rt["high"]
                            if rt.get("low"):
                                price_df.loc[price_df.index[-1], "low"] = rt["low"]
                            if rt.get("volume"):
                                price_df.loc[price_df.index[-1], "volume"] = rt["volume"]
                            is_intraday = True

                enriched = compute_screener_indicators(price_df)
                diag = diagnose_h_strategy(enriched)

                # 取得名稱
                name = rt.get("name", sid) if rt else sid
                if name == sid and "stock_name" in price_df.columns:
                    sn = price_df["stock_name"].iloc[-1]
                    if pd.notna(sn):
                        name = str(sn)

                close_val = diag.get("summary", {}).get("close")
                item = {
                    "stock_id": sid, "name": name,
                    "close": close_val,
                    "diagnose": diag,
                }
                if rt_time:
                    item["rt_time"] = rt_time
                results.append(item)

            except Exception as e:
                print(f"H策略診斷 {sid} 失敗：{e}")
                results.append({
                    "stock_id": sid, "name": sid,
                    "close": None,
                    "diagnose": {
                        "passed": False, "passed_count": 0,
                        "total_checks": 9, "checks": [], "summary": {},
                        "error": f"診斷失敗：{e}",
                    },
                })

        _h_diagnose_status["result"] = _sanitize_for_json({
            "date": today_str, "results": results, "is_intraday": is_intraday,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        _h_diagnose_status["error"] = str(e)
    finally:
        _h_diagnose_status["done"] = True
        _h_diagnose_status["running"] = False


@app.route("/api/h-diagnose/run", methods=["POST"])
def api_h_diagnose_run():
    global _h_diagnose_status
    if _h_diagnose_status.get("running"):
        return jsonify({"started": True, "msg": "已在執行中"})

    data = request.json or {}
    stock_ids = data.get("stock_ids", [])
    if not stock_ids:
        return jsonify({"error": "請提供股票代號"}), 400

    _h_diagnose_status = {
        "running": False, "done": False, "error": None,
        "progress": 0, "total": len(stock_ids), "current": "啟動中...",
        "result": None,
    }
    t = threading.Thread(target=_run_h_diagnose_bg, args=(stock_ids,), daemon=True)
    t.start()
    return jsonify({"started": True})


@app.route("/api/h-diagnose/status", methods=["GET"])
def api_h_diagnose_status():
    return jsonify({
        "running": _h_diagnose_status["running"],
        "done": _h_diagnose_status["done"],
        "progress": _h_diagnose_status["progress"],
        "total": _h_diagnose_status["total"],
        "current": _h_diagnose_status["current"],
        "error": _h_diagnose_status["error"],
        "result": _h_diagnose_status["result"],
    })


# ═══════════════════════════════════════════════════════════
# 啟動
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os, socket

    from config import AI_PROVIDER, GEMINI_API_KEY

    if not FINMIND_TOKEN:
        print("⚠️  未設定 FINMIND_TOKEN（環境變數），部分功能無法使用")

    # AI 模組狀態
    if AI_PROVIDER == "gemini" and GEMINI_API_KEY:
        print("🤖 AI 分析：Gemini（gemini-2.0-flash）")
    elif AI_PROVIDER == "claude" and ANTHROPIC_API_KEY:
        print("🤖 AI 分析：Claude（claude-sonnet-4-6）")
    elif ANTHROPIC_API_KEY:
        print("🤖 AI 分析：Claude（claude-sonnet-4-6）← 未設定指定 provider 的 key，自動退回")
    elif GEMINI_API_KEY:
        print("🤖 AI 分析：Gemini（gemini-2.0-flash）← 未設定指定 provider 的 key，自動退回")
    else:
        print("⚠️  未設定任何 AI API Key，將使用規則式分析")

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    port = int(os.environ.get("PORT", 5001))
    print("=" * 50)
    print("  台股覆盤工具 v1")
    print("=" * 50)
    print(f"  本機：http://localhost:{port}")
    print(f"  區網：http://{local_ip}:{port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
