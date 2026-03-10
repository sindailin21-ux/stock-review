"""
data_fetcher.py
從 FinMind API 與 TWSE/TPEX 公開資料抓取台股資料
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from config import FINMIND_TOKEN, DATA_DAYS


BASE_URL = "https://api.finmindtrade.com/api/v4/data"


def _get_start_date(days=DATA_DAYS):
    """取得資料起始日期"""
    return (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")


def _fetch(dataset, stock_id, start_date=None, extra_params=None):
    """通用 API 請求函式"""
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date or _get_start_date(),
        "token": FINMIND_TOKEN,
    }
    if extra_params:
        params.update(extra_params)

    try:
        resp = requests.get(BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 200 and data.get("data"):
            return pd.DataFrame(data["data"])
        else:
            print(f"  [警告] {stock_id} / {dataset}：{data.get('msg', '無資料')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  [錯誤] {stock_id} / {dataset}：{e}")
        return pd.DataFrame()


# ── 全市場清單與日行情 ─────────────────────────────────────

def fetch_stock_list():
    """
    抓取全市場上市櫃股票清單（TaiwanStockInfo）。
    過濾條件：
    - type in ('twse', 'tpex')：排除興櫃
    - stock_id 為 4 位數字（1xxx-9xxx）：排除 ETF(00xx)、權證、存託憑證
    """
    try:
        resp = requests.get(
            BASE_URL,
            params={"dataset": "TaiwanStockInfo", "token": FINMIND_TOKEN},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == 200 and data.get("data"):
            df = pd.DataFrame(data["data"])
            # 過濾上市+上櫃
            if "type" in df.columns:
                df = df[df["type"].isin(["twse", "tpex"])].copy()
            # 只保留 4 位數字代號（一般股票），排除 ETF(00xx)、權證等
            df = df[df["stock_id"].str.match(r"^[1-9]\d{3}$")].copy()
            return df.reset_index(drop=True)
        else:
            print(f"  [警告] TaiwanStockInfo：{data.get('msg', '無資料')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"  [錯誤] TaiwanStockInfo：{e}")
        return pd.DataFrame()


def fetch_market_daily(date=None):
    """
    從 TWSE + TPEX 公開資料抓取全市場日行情（不需 FinMind 付費帳號）。
    date 格式 "YYYY-MM-DD"，預設為最新資料。
    回傳 DataFrame 含 stock_id, name, exchange, Trading_Volume(股), Trading_Money(元) 欄位。
    exchange 值為 'twse' 或 'tpex'，供後續判斷該用哪個 API。
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    rows = []

    # ── TWSE 上市 ──
    try:
        twse_params = {"response": "json"}
        if date:
            twse_params["date"] = date.replace("-", "")
        resp = requests.get(
            "https://www.twse.com.tw/exchangeReport/STOCK_DAY_ALL",
            params=twse_params, timeout=30, headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        twse_count = 0
        # fields: 證券代號, 證券名稱, 成交股數, 成交金額, 開盤價, 最高價, 最低價, 收盤價, 漲跌價差, 成交筆數
        for row in data.get("data", []):
            try:
                rows.append({
                    "stock_id": row[0].strip(),
                    "name": row[1].strip(),
                    "exchange": "twse",
                    "Trading_Volume": int(row[2].replace(",", "")),
                    "Trading_Money": int(row[3].replace(",", "")),
                })
                twse_count += 1
            except (ValueError, IndexError):
                continue
        print(f"  TWSE 上市：{twse_count} 檔")
    except Exception as e:
        print(f"  [錯誤] TWSE 上市行情：{e}")

    # ── TPEX 上櫃（需精確日期，若當日無資料則往回找最近交易日）──
    tpex_ok = False
    try:
        if date:
            dt = datetime.strptime(date, "%Y-%m-%d")
        else:
            dt = datetime.today()

        otc_count = 0
        for lookback in range(5):  # 最多往回找 5 天
            try_dt = dt - timedelta(days=lookback)
            tw_date = f"{try_dt.year - 1911}/{try_dt.month:02d}/{try_dt.day:02d}"

            resp = requests.get(
                "https://www.tpex.org.tw/web/stock/aftertrading/otc_quotes_no1430/stk_wn1430_result.php",
                params={"l": "zh-tw", "d": tw_date, "se": "EW"},
                timeout=30, headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            for table in data.get("tables", []):
                for row in table.get("data", []):
                    try:
                        rows.append({
                            "stock_id": row[0].strip(),
                            "name": row[1].strip(),
                            "exchange": "tpex",
                            "Trading_Volume": int(row[7].replace(",", "")),
                            "Trading_Money": int(row[8].replace(",", "")),
                        })
                        otc_count += 1
                    except (ValueError, IndexError):
                        continue

            if otc_count > 0:
                print(f"  TPEX 上櫃：{otc_count} 檔（{tw_date}）")
                tpex_ok = True
                break
        else:
            print(f"  TPEX 上櫃：0 檔（近 5 天皆無資料）")
    except Exception as e:
        print(f"  [錯誤] TPEX 上櫃行情：{e}，嘗試 FinMind fallback...")

    # ── TPEX fallback：改用 FinMind TaiwanStockPrice ──
    if not tpex_ok and FINMIND_TOKEN:
        try:
            query_date = date or datetime.today().strftime("%Y-%m-%d")
            fm_resp = requests.get(BASE_URL, params={
                "dataset": "TaiwanStockPrice",
                "start_date": query_date,
                "end_date": query_date,
                "token": FINMIND_TOKEN,
            }, timeout=30)
            fm_resp.raise_for_status()
            fm_data = fm_resp.json()
            if fm_data.get("status") == 200 and fm_data.get("data"):
                # 取得上櫃股清單以過濾
                otc_ids = set()
                try:
                    info_resp = requests.get(BASE_URL, params={
                        "dataset": "TaiwanStockInfo",
                        "token": FINMIND_TOKEN,
                    }, timeout=30)
                    if info_resp.status_code == 200:
                        for item in info_resp.json().get("data", []):
                            if item.get("type") == "tpex":
                                otc_ids.add(item["stock_id"])
                except Exception:
                    pass

                otc_count = 0
                for item in fm_data["data"]:
                    sid = item.get("stock_id", "")
                    # 如果有上櫃清單就用，沒有就用股號猜測
                    if otc_ids:
                        if sid not in otc_ids:
                            continue
                    else:
                        if not sid[:1].isdigit() or len(sid) != 4:
                            continue
                    rows.append({
                        "stock_id": sid,
                        "name": item.get("stock_name", sid),
                        "exchange": "tpex",
                        "Trading_Volume": int(item.get("Trading_Volume", 0)),
                        "Trading_Money": int(item.get("Trading_Money", 0)),
                    })
                    otc_count += 1
                print(f"  TPEX 上櫃（FinMind fallback）：{otc_count} 檔")
        except Exception as e:
            print(f"  [錯誤] FinMind TPEX fallback：{e}")

    if not rows:
        print("  [警告] 全市場行情：無資料（可能為非交易日）")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(f"  全市場合計：{len(df)} 檔")
    return df


# ── 輔助：安全數值轉換 ──────────────────────────────────────

_PUB_HEADERS = {"User-Agent": "Mozilla/5.0"}

def _safe_float(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None

def _safe_int(s):
    try:
        return int(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None


# ── 單一股票歷史行情（TWSE / TPEX 公開資料）────────────────

def _parse_roc_date(roc_str):
    """將民國日期 '115/02/17' 轉為 'YYYY-MM-DD'（自動去除 * 等註記符號）"""
    import re
    cleaned = re.sub(r"[^\d/]", "", roc_str.strip())
    parts = cleaned.split("/")
    y = int(parts[0]) + 1911
    return f"{y}-{parts[1]}-{parts[2]}"


def _fetch_twse_monthly(stock_id, yyyymmdd):
    """
    呼叫 TWSE STOCK_DAY 取得單月 OHLCV。
    回傳 list[dict]，每筆 {date, open, high, low, close, volume(股)}。
    """
    rows = []
    try:
        resp = requests.get(
            "https://www.twse.com.tw/exchangeReport/STOCK_DAY",
            params={"response": "json", "date": yyyymmdd, "stockNo": stock_id},
            timeout=15, headers=_PUB_HEADERS,
        )
        if resp.status_code != 200:
            return rows
        data = resp.json()
        if data.get("stat") != "OK":
            return rows
        # fields: [0]日期, [1]成交股數, [2]成交金額, [3]開盤價, [4]最高價, [5]最低價, [6]收盤價
        for row in data.get("data", []):
            try:
                c = _safe_float(row[6])
                v = _safe_int(row[1])
                if c is None or v is None:
                    continue
                rows.append({
                    "date": _parse_roc_date(row[0]),
                    "open":  _safe_float(row[3]) or c,
                    "high":  _safe_float(row[4]) or c,
                    "low":   _safe_float(row[5]) or c,
                    "close": c,
                    "volume": v,
                })
            except (IndexError, KeyError, ValueError):
                continue
    except Exception:
        pass
    return rows


def _fetch_tpex_monthly(stock_id, yyyy_mm_dd):
    """
    呼叫 TPEX 新版 API 取得單月 OHLCV。
    yyyy_mm_dd 格式: 'YYYY/MM/DD'（西元，任一天即可，回傳整月）。
    回傳 list[dict]，每筆 {date, open, high, low, close, volume(股)}。
    注意：TPEX 成交張數 × 1000 = 股數。
    403 時 fallback 到 FinMind TaiwanStockPrice。
    """
    rows = []
    try:
        resp = requests.get(
            "https://www.tpex.org.tw/www/zh-tw/afterTrading/tradingStock",
            params={"date": yyyy_mm_dd, "code": stock_id, "response": "json"},
            timeout=15, headers=_PUB_HEADERS,
        )
        if resp.status_code == 403:
            raise requests.exceptions.HTTPError("403 Forbidden")
        if resp.status_code != 200:
            return rows
        data = resp.json()
        if data.get("stat") != "ok" or not data.get("tables"):
            return rows
        # fields: [0]日期(ROC), [1]成交張數, [2]成交仟元, [3]開盤, [4]最高, [5]最低, [6]收盤
        for row in data["tables"][0].get("data", []):
            try:
                c = _safe_float(row[6])
                v = _safe_int(row[1])
                if c is None or v is None:
                    continue
                rows.append({
                    "date": _parse_roc_date(row[0]),
                    "open":  _safe_float(row[3]) or c,
                    "high":  _safe_float(row[4]) or c,
                    "low":   _safe_float(row[5]) or c,
                    "close": c,
                    "volume": v * 1000,  # 張 → 股
                })
            except (IndexError, KeyError, ValueError):
                continue
    except Exception:
        # Fallback: FinMind TaiwanStockPrice
        if FINMIND_TOKEN:
            try:
                # 從 yyyy_mm_dd ('YYYY/MM/DD') 解析月份範圍
                parts = yyyy_mm_dd.split("/")
                y, m = int(parts[0]), int(parts[1])
                import calendar
                _, last_day = calendar.monthrange(y, m)
                start = f"{y}-{m:02d}-01"
                end = f"{y}-{m:02d}-{last_day:02d}"
                fm_resp = requests.get(BASE_URL, params={
                    "dataset": "TaiwanStockPrice",
                    "data_id": stock_id,
                    "start_date": start,
                    "end_date": end,
                    "token": FINMIND_TOKEN,
                }, timeout=15)
                if fm_resp.status_code == 200:
                    fm_data = fm_resp.json()
                    if fm_data.get("status") == 200:
                        for item in fm_data.get("data", []):
                            rows.append({
                                "date": item["date"],
                                "open":  float(item.get("open", 0)),
                                "high":  float(item.get("max", 0)),
                                "low":   float(item.get("min", 0)),
                                "close": float(item.get("close", 0)),
                                "volume": int(item.get("Trading_Volume", 0)),
                            })
            except Exception:
                pass
    return rows


def fetch_stock_price_public(stock_id, exchange="twse", target_date=None, months=5):
    """
    從 TWSE 或 TPEX 公開 API 抓取單一股票 N 個月的日 K 線。
    exchange: 'twse' 或 'tpex'
    回傳 DataFrame(date, open, high, low, close, volume)。
    """
    import time as _time

    if target_date:
        end = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        end = datetime.today()

    all_rows = []
    for m in range(months):
        # 計算目標月份（從 end 往回推 m 個月，用月份算術避免跳月）
        y = end.year
        mo = end.month - m
        while mo <= 0:
            mo += 12
            y -= 1
        dt = datetime(y, mo, 1)
        if exchange == "tpex":
            date_param = f"{dt.year}/{dt.month:02d}/01"
            rows = _fetch_tpex_monthly(stock_id, date_param)
        else:
            date_param = f"{dt.year}{dt.month:02d}01"
            rows = _fetch_twse_monthly(stock_id, date_param)
        all_rows.extend(rows)
        _time.sleep(0.3)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    # 去重（跨月邊界可能重複）+ 只取 target_date 之前的資料
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    if target_date:
        df = df[df["date"] <= target_date].copy()
    return df


def fetch_stock_prices_batch(stock_ids, exchange_map, target_date=None,
                             months=5, status=None):
    """
    批次抓取多支股票的歷史 OHLCV（多線程並行 TWSE/TPEX 公開 API）。
    exchange_map: dict[stock_id] → 'twse' 或 'tpex'
    回傳 dict[stock_id] → DataFrame(date, open, high, low, close, volume)。
    使用 ThreadPoolExecutor(max_workers=10) 加速，預估 20 分鐘 → 2-3 分鐘。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    result = {}
    total = len(stock_ids)
    counter = {"done": 0}
    lock = threading.Lock()

    def _worker(sid):
        """單支股票抓取（工作線程）"""
        if status and status.get("cancel"):
            return sid, None

        exchange = exchange_map.get(sid, "twse")
        df = fetch_stock_price_public(sid, exchange, target_date, months)

        # Thread-safe 進度更新
        with lock:
            counter["done"] += 1
            cnt = counter["done"]

        if status:
            status["current"] = f"載入歷史行情：{sid}（{cnt}/{total}）"

        if cnt % 100 == 0:
            print(f"  📊 歷史行情進度：{cnt}/{total}")

        return sid, df if not df.empty else None

    print(f"📊 開始多線程抓取歷史行情：{total} 檔（10 threads）")

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_map = {executor.submit(_worker, sid): sid for sid in stock_ids}
        for future in as_completed(future_map):
            try:
                sid, df = future.result()
                if df is not None:
                    result[sid] = df
            except Exception as e:
                print(f"  [錯誤] 歷史行情抓取：{e}")

            # 取消檢查
            if status and status.get("cancel"):
                for f in future_map:
                    f.cancel()
                break

    print(f"📊 歷史行情完成：{len(result)} / {total} 檔 × {months} 月")
    return result


# ── 全市場法人籌碼（TWSE T86 + TPEX，批次）──────────────────

def fetch_institutional_batch(target_date=None, days=5, status=None):
    """
    從 TWSE (T86) + TPEX 公開資料，批次取得近 N 天全市場法人買賣超。
    回傳 dict[stock_id] → pd.DataFrame(date, Foreign_Investor_Buy,
        Foreign_Investor_Sell, Investment_Trust_Buy, Investment_Trust_Sell)
    單位：股。
    """
    import time as _time

    if target_date:
        end = datetime.strptime(target_date, "%Y-%m-%d")
    else:
        end = datetime.today()

    all_records = {}
    days_collected = 0
    current = end
    max_back = days * 3 + 10

    while days_collected < days and (end - current).days < max_back:
        if current.weekday() >= 5:
            current -= timedelta(days=1)
            continue

        if status and status.get("cancel"):
            break

        if status:
            status["current"] = f"載入法人籌碼 {current.strftime('%m/%d')}（{days_collected}/{days}）"

        date_twse = current.strftime("%Y%m%d")
        tw_y = current.year - 1911
        date_tpex = f"{tw_y}/{current.month:02d}/{current.day:02d}"
        date_str = current.strftime("%Y-%m-%d")
        day_has_data = False

        # ── TWSE T86 ──
        # fields[2]=外資買, [3]=外資賣, [8]=投信買, [9]=投信賣
        try:
            resp = requests.get(
                "https://www.twse.com.tw/fund/T86",
                params={"response": "json", "date": date_twse, "selectType": "ALLBUT0999"},
                timeout=20, headers=_PUB_HEADERS,
            )
            if resp.status_code == 200:
                for row in resp.json().get("data", []):
                    try:
                        sid = row[0].strip()
                        all_records.setdefault(sid, []).append({
                            "date": date_str,
                            "Foreign_Investor_Buy":  _safe_int(row[2]) or 0,
                            "Foreign_Investor_Sell": _safe_int(row[3]) or 0,
                            "Investment_Trust_Buy":  _safe_int(row[8]) or 0,
                            "Investment_Trust_Sell": _safe_int(row[9]) or 0,
                        })
                        day_has_data = True
                    except (IndexError, KeyError):
                        continue
        except Exception as e:
            print(f"  [TWSE T86] {date_str}: {e}")

        _time.sleep(0.4)

        # ── TPEX 法人 ──
        # fields: 代號(0), 名稱(1), 外資買(2), 外資賣(3), 外資超(4),
        #         投信買(5), 投信賣(6), 投信超(7), ...
        tpex_inst_ok = False
        try:
            resp = requests.get(
                "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php",
                params={"l": "zh-tw", "d": date_tpex, "se": "EW", "t": "D"},
                timeout=20, headers=_PUB_HEADERS,
            )
            if resp.status_code == 403:
                raise requests.exceptions.HTTPError("403 Forbidden")
            if resp.status_code == 200:
                for table in resp.json().get("tables", []):
                    for row in table.get("data", []):
                        try:
                            sid = row[0].strip()
                            all_records.setdefault(sid, []).append({
                                "date": date_str,
                                "Foreign_Investor_Buy":  _safe_int(row[2]) or 0,
                                "Foreign_Investor_Sell": _safe_int(row[3]) or 0,
                                "Investment_Trust_Buy":  _safe_int(row[5]) or 0,
                                "Investment_Trust_Sell": _safe_int(row[6]) or 0,
                            })
                            if not day_has_data:
                                day_has_data = True
                        except (IndexError, KeyError):
                            continue
                tpex_inst_ok = True
        except Exception as e:
            print(f"  [TPEX法人] {date_str}: {e}")

        # ── TPEX 法人 fallback：FinMind ──
        if not tpex_inst_ok and FINMIND_TOKEN:
            try:
                fm_resp = requests.get(BASE_URL, params={
                    "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
                    "start_date": date_str,
                    "end_date": date_str,
                    "token": FINMIND_TOKEN,
                }, timeout=20)
                if fm_resp.status_code == 200:
                    fm_data = fm_resp.json()
                    if fm_data.get("status") == 200:
                        for item in fm_data.get("data", []):
                            sid = item.get("stock_id", "")
                            name = item.get("name", "")
                            buy = int(item.get("buy", 0))
                            sell = int(item.get("sell", 0))
                            if name == "Foreign_Investor":
                                all_records.setdefault(sid, []).append({
                                    "date": date_str,
                                    "Foreign_Investor_Buy": buy,
                                    "Foreign_Investor_Sell": sell,
                                    "Investment_Trust_Buy": 0,
                                    "Investment_Trust_Sell": 0,
                                })
                            elif name == "Investment_Trust":
                                # 合併到同一筆
                                existing = None
                                for rec in all_records.get(sid, []):
                                    if rec["date"] == date_str:
                                        existing = rec
                                        break
                                if existing:
                                    existing["Investment_Trust_Buy"] = buy
                                    existing["Investment_Trust_Sell"] = sell
                                else:
                                    all_records.setdefault(sid, []).append({
                                        "date": date_str,
                                        "Foreign_Investor_Buy": 0,
                                        "Foreign_Investor_Sell": 0,
                                        "Investment_Trust_Buy": buy,
                                        "Investment_Trust_Sell": sell,
                                    })
                        if not day_has_data:
                            day_has_data = True
                        print(f"  [TPEX法人] {date_str}: FinMind fallback OK")
            except Exception as e2:
                print(f"  [TPEX法人 FinMind fallback] {date_str}: {e2}")

        if day_has_data:
            days_collected += 1

        current -= timedelta(days=1)
        _time.sleep(0.4)

    # ── 轉換為 per-stock DataFrames ──
    result = {}
    for sid, records in all_records.items():
        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        result[sid] = df

    print(f"🏦 法人籌碼：{len(result)} 檔 × {days_collected} 交易日")
    return result


# ── 技術面 ──────────────────────────────────────────────────

def fetch_price(stock_id):
    """抓取日 K 線資料（開高低收量）"""
    df = _fetch("TaiwanStockPrice", stock_id)
    if df.empty:
        return df
    df = df[["date", "open", "max", "min", "close", "Trading_Volume"]].copy()
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── 即時報價（TWSE MIS API）──────────────────────────────────

def fetch_realtime_quote(stock_id: str):
    """
    從 TWSE MIS API 取得即時報價（盤中 9:00-13:30 有效）。
    自動判斷上市(tse)/上櫃(otc)。
    回傳 dict: {price, open, high, low, yesterday, volume, time, date, name}
    若無法取得回傳 None。
    """
    for ex in ("tse", "otc"):
        try:
            resp = requests.get(
                "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                params={"ex_ch": f"{ex}_{stock_id}.tw"},
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
            data = resp.json()
            arr = data.get("msgArray", [])
            if not arr:
                continue
            s = arr[0]
            price = s.get("z", "-")
            if price == "-":
                # 試 match price（五檔揭示成交價），盤前可能 z 為 "-"
                price = s.get("l", "-")  # 用最低價當替代
            if price == "-":
                continue
            return {
                "price": float(price),
                "open": float(s["o"]) if s.get("o") and s["o"] != "-" else None,
                "high": float(s["h"]) if s.get("h") and s["h"] != "-" else None,
                "low": float(s["l"]) if s.get("l") and s["l"] != "-" else None,
                "yesterday": float(s["y"]) if s.get("y") and s["y"] != "-" else None,
                "volume": int(s["v"]) * 1000 if s.get("v") else 0,  # 張→股
                "time": s.get("t", ""),
                "date": s.get("d", ""),
                "name": s.get("n", stock_id),
                "exchange": ex,
            }
        except Exception:
            continue
    return None


def fetch_realtime_quotes_batch(stock_ids: list, exchange_map: dict = None,
                                batch_size: int = 50) -> dict:
    """
    批次取得多支股票即時報價（TWSE MIS API）。
    exchange_map: {stock_id → 'twse'/'tpex'}，若無則自動嘗試兩種。
    回傳 dict[stock_id] → {price, open, high, low, yesterday, volume, time, date, name}
    """
    import time as _time

    results = {}
    # 分組：上市 vs 上櫃
    tse_ids = []
    otc_ids = []
    unknown_ids = []

    for sid in stock_ids:
        if exchange_map:
            ex = exchange_map.get(sid, "")
            if ex == "twse":
                tse_ids.append(sid)
            elif ex == "tpex":
                otc_ids.append(sid)
            else:
                unknown_ids.append(sid)
        else:
            unknown_ids.append(sid)

    def _batch_fetch(ids, exchange):
        """批次抓取同一交易所的即時報價"""
        for i in range(0, len(ids), batch_size):
            chunk = ids[i:i + batch_size]
            ex_ch = "|".join(f"{exchange}_{sid}.tw" for sid in chunk)
            try:
                resp = requests.get(
                    "https://mis.twse.com.tw/stock/api/getStockInfo.jsp",
                    params={"ex_ch": ex_ch},
                    timeout=15,
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                resp.raise_for_status()
                data = resp.json()
                for s in data.get("msgArray", []):
                    # 從 ex_ch 欄位解析股票代碼（格式：tse_2330.tw）
                    code = s.get("c", "")
                    if not code:
                        ch = s.get("ch", "")
                        code = ch.split("_")[-1].replace(".tw", "") if "_" in ch else ch.replace(".tw", "")
                    price = s.get("z", "-")
                    if price == "-":
                        price = s.get("l", "-")
                    if price == "-":
                        continue
                    results[code] = {
                        "price": float(price),
                        "open": float(s["o"]) if s.get("o") and s["o"] != "-" else None,
                        "high": float(s["h"]) if s.get("h") and s["h"] != "-" else None,
                        "low": float(s["l"]) if s.get("l") and s["l"] != "-" else None,
                        "yesterday": float(s["y"]) if s.get("y") and s["y"] != "-" else None,
                        "volume": int(s["v"]) * 1000 if s.get("v") else 0,
                        "time": s.get("t", ""),
                        "date": s.get("d", ""),
                        "name": s.get("n", code),
                        "exchange": exchange,
                    }
            except Exception as e:
                print(f"   ⚠️ 即時報價批次抓取失敗（{exchange}）：{e}")
            _time.sleep(0.3)

    if tse_ids:
        _batch_fetch(tse_ids, "tse")
    if otc_ids:
        _batch_fetch(otc_ids, "otc")
    # 未知交易所：先試 tse 再試 otc
    if unknown_ids:
        _batch_fetch(unknown_ids, "tse")
        missing = [sid for sid in unknown_ids if sid not in results]
        if missing:
            _batch_fetch(missing, "otc")

    return results


# ── 籌碼面 ──────────────────────────────────────────────────

def fetch_institutional(stock_id):
    """抓取三大法人買賣超（外資、投信、自營商）"""
    df = _fetch("TaiwanStockInstitutionalInvestorsBuySell", stock_id)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_margin(stock_id):
    """抓取融資融券資料"""
    df = _fetch("TaiwanStockMarginPurchaseShortSale", stock_id)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── 基本面 ──────────────────────────────────────────────────

def fetch_revenue(stock_id):
    """抓取月營收資料（取 14 個月以便計算年增率）"""
    start = (datetime.today() - timedelta(days=450)).strftime("%Y-%m-%d")
    df = _fetch("TaiwanStockMonthRevenue", stock_id, start_date=start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def fetch_eps(stock_id):
    """抓取每季 EPS"""
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    df = _fetch("TaiwanStockFinancialStatements", stock_id, start_date=start)
    if df.empty:
        return df
    # 只取 EPS 欄位
    eps_df = df[df["type"] == "EPS"].copy() if "type" in df.columns else df
    eps_df["date"] = pd.to_datetime(eps_df["date"])
    eps_df = eps_df.sort_values("date").reset_index(drop=True)
    return eps_df


# ── 產業分類對照表 ─────────────────────────────────────────

_industry_cache = None
_industry_cache_time = 0


def fetch_industry_map() -> dict:
    """
    從 TWSE / TPEX 公開網站抓取產業分類。
    回傳 dict[stock_id] → industry_name（如 "半導體業"）。
    失敗回傳空 dict（不影響其他功能）。
    結果快取 1 小時，避免重複呼叫。
    """
    global _industry_cache, _industry_cache_time
    import time as _time
    if _industry_cache is not None and (_time.time() - _industry_cache_time) < 3600:
        return _industry_cache
    industry = {}

    # ── 上市（strMode=2）──
    try:
        resp = requests.get(
            "https://isin.twse.com.tw/isin/C_public.jsp",
            params={"strMode": "2"},
            timeout=30,
            headers=_PUB_HEADERS,
        )
        resp.encoding = "big5"
        tables = pd.read_html(resp.text, header=0)
        if tables:
            df = tables[0]
            # 表格格式：第一欄 "有價證券代號及名稱"（如 "2330　台積電"），產業別欄
            col0 = df.columns[0]       # "有價證券代號及名稱"
            # 找到「產業別」欄（可能叫 "產業別" 或 "CFICode" 旁邊）
            ind_col = None
            for c in df.columns:
                if "產業別" in str(c):
                    ind_col = c
                    break
            if ind_col:
                for _, row in df.iterrows():
                    try:
                        code_name = str(row[col0]).strip()
                        # 格式 "2330　台積電" 或 "2330 台積電"
                        parts = code_name.replace("\u3000", " ").split()
                        if parts and len(parts[0]) == 4 and parts[0].isdigit():
                            sid = parts[0]
                            ind = str(row[ind_col]).strip()
                            if ind and ind != "nan":
                                industry[sid] = ind
                    except Exception:
                        continue
        print(f"  📂 上市產業分類：{len(industry)} 檔")
    except Exception as e:
        print(f"  [警告] 上市產業分類抓取失敗：{e}")

    # ── 上櫃（strMode=4）──
    otc_count = 0
    try:
        resp = requests.get(
            "https://isin.twse.com.tw/isin/C_public.jsp",
            params={"strMode": "4"},
            timeout=30,
            headers=_PUB_HEADERS,
        )
        resp.encoding = "big5"
        tables = pd.read_html(resp.text, header=0)
        if tables:
            df = tables[0]
            col0 = df.columns[0]
            ind_col = None
            for c in df.columns:
                if "產業別" in str(c):
                    ind_col = c
                    break
            if ind_col:
                for _, row in df.iterrows():
                    try:
                        code_name = str(row[col0]).strip()
                        parts = code_name.replace("\u3000", " ").split()
                        if parts and len(parts[0]) == 4 and parts[0].isdigit():
                            sid = parts[0]
                            ind = str(row[ind_col]).strip()
                            if ind and ind != "nan":
                                industry[sid] = ind
                                otc_count += 1
                    except Exception:
                        continue
        print(f"  📂 上櫃產業分類：{otc_count} 檔")
    except Exception as e:
        print(f"  [警告] 上櫃產業分類抓取失敗：{e}")

    print(f"  📂 產業分類合計：{len(industry)} 檔")
    _industry_cache = industry
    _industry_cache_time = _time.time()
    return industry


# ── 整合取得單一股票所有資料 ────────────────────────────────

def fetch_all(stock_id):
    """一次抓取單一股票所有需要的資料"""
    print(f"  抓取 {stock_id} 資料中...")
    return {
        "price": fetch_price(stock_id),
        "institutional": fetch_institutional(stock_id),
        "margin": fetch_margin(stock_id),
        "revenue": fetch_revenue(stock_id),
        "eps": fetch_eps(stock_id),
    }
