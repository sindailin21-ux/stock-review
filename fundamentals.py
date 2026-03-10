"""
fundamentals.py
整理籌碼面與基本面資料，輸出摘要供 AI 分析
"""

import pandas as pd


def get_institutional_summary(df):
    """
    整理三大法人最新買賣超
    回傳最近 5 日的外資、投信、自營商買賣超
    """
    if df.empty:
        return {}

    summary = {}

    # 依法人類型分組
    for investor_type in ["Foreign_Investor", "Investment_Trust", "Dealer"]:
        type_map = {
            "Foreign_Investor": "外資",
            "Investment_Trust": "投信",
            "Dealer": "自營商",
        }
        name = type_map.get(investor_type, investor_type)

        sub = df[df["name"] == investor_type].copy() if "name" in df.columns else pd.DataFrame()
        if sub.empty:
            # 嘗試其他欄位名稱（FinMind 格式可能不同）
            continue

        sub = sub.sort_values("date").tail(5)
        if "buy" in sub.columns and "sell" in sub.columns:
            sub["net"] = sub["buy"] - sub["sell"]
            # FinMind 單位是股，除以 1000 換算為張
            recent_net = int(sub["net"].iloc[-1] // 1000) if not sub.empty else 0
            five_day_net = int(sub["net"].sum() // 1000)
            summary[f"{name}_今日(張)"] = recent_net
            summary[f"{name}_5日累計(張)"] = five_day_net

    # 若 FinMind 回傳的是整合格式（有欄位 buy/sell per type）
    if not summary and not df.empty:
        df_latest = df.sort_values("date").tail(15)
        for col_buy, col_sell, label in [
            ("Foreign_Investor_Buy", "Foreign_Investor_Sell", "外資"),
            ("Investment_Trust_Buy", "Investment_Trust_Sell", "投信"),
            ("Dealer_Buy", "Dealer_Sell", "自營商"),
        ]:
            if col_buy in df_latest.columns:
                # FinMind 單位是股，除以 1000 換算為張
                net_today = int((df_latest[col_buy].iloc[-1] - df_latest[col_sell].iloc[-1]) // 1000)
                net_5d = int((df_latest[col_buy] - df_latest[col_sell]).tail(5).sum() // 1000)
                summary[f"{label}_今日(張)"] = net_today
                summary[f"{label}_5日累計(張)"] = net_5d

    return summary


def get_margin_summary(df):
    """整理融資融券資料"""
    if df.empty:
        return {}

    df = df.sort_values("date").tail(5)
    summary = {}

    # 嘗試常見欄位名稱
    for col, label in [
        ("MarginPurchaseBuy", "融資買進"),
        ("MarginPurchaseSell", "融資賣出"),
        ("ShortSaleBuy", "融券買進"),
        ("ShortSaleSell", "融券賣出"),
        ("MarginPurchaseBalance", "融資餘額"),
        ("ShortSaleBalance", "融券餘額"),
    ]:
        if col in df.columns:
            summary[label] = int(df[col].iloc[-1])

    return summary


def get_revenue_summary(df):
    """
    整理月營收：最近月份、月增率、年增率
    FinMind 的 revenue_year / revenue_month 是「年份/月份」數字，
    不是去年同期營收，需自行比對同月資料計算 YoY。
    """
    if df.empty:
        return {}

    df_sorted = df.sort_values("date").copy()

    summary = {}
    if "revenue" not in df_sorted.columns or len(df_sorted) < 1:
        return summary

    latest = df_sorted.iloc[-1]
    summary["最新月份"] = str(latest.get("date", ""))[:7]
    # FinMind 回傳單位為「元」，轉換為億元方便閱讀
    rev_raw = float(latest.get("revenue", 0))
    summary["月營收(億元)"] = round(rev_raw / 1e8, 2)

    # 月增率（與前一個月比）
    if len(df_sorted) >= 2:
        prev = df_sorted.iloc[-2]
        mom = ((latest["revenue"] - prev["revenue"]) / prev["revenue"] * 100) if prev["revenue"] else 0
        summary["月增率(%)"] = round(float(mom), 2)

    # 年增率（比對去年同月）
    if "revenue_month" in df_sorted.columns and "revenue_year" in df_sorted.columns:
        cur_month = int(latest["revenue_month"])
        cur_year = int(latest["revenue_year"])
        # 找去年同月
        same_month_last_year = df_sorted[
            (df_sorted["revenue_month"].astype(int) == cur_month) &
            (df_sorted["revenue_year"].astype(int) == cur_year - 1)
        ]
        if not same_month_last_year.empty:
            prev_rev = float(same_month_last_year.iloc[-1]["revenue"])
            if prev_rev > 0:
                yoy = (rev_raw - prev_rev) / prev_rev * 100
                if abs(yoy) <= 1000:
                    summary["年增率(%)"] = round(yoy, 2)

    return summary


def get_eps_summary(df):
    """整理最近 4 季 EPS"""
    if df.empty:
        return {}

    summary = {}
    df = df.sort_values("date").tail(4)

    if "value" in df.columns:
        eps_list = df[["date", "value"]].copy()
        eps_list["date"] = eps_list["date"].astype(str).str[:7]
        eps_records = []
        for _, row in eps_list.iterrows():
            eps_records.append(f"{row['date']}: {round(float(row['value']), 2)}")
        summary["近4季EPS"] = " / ".join(eps_records)
        summary["近4季合計EPS"] = round(float(df["value"].sum()), 2)

    return summary


def get_full_fundamental_summary(raw_data):
    """
    整合所有基本面與籌碼面資料
    raw_data = fetch_all() 的回傳值
    """
    return {
        "institutional": get_institutional_summary(raw_data.get("institutional", pd.DataFrame())),
        "margin": get_margin_summary(raw_data.get("margin", pd.DataFrame())),
        "revenue": get_revenue_summary(raw_data.get("revenue", pd.DataFrame())),
        "eps": get_eps_summary(raw_data.get("eps", pd.DataFrame())),
    }
