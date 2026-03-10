"""
portfolio_store.py
記憶體型持股清單管理

資料格式：
  [{"stock_id": "2330", "name": "台積電", "shares": 1000, "cost_price": 850.0}, ...]

注意：
  - 伺服器重啟後資料清空，請定期從 web 匯出 CSV 備份
  - 未來可擴充為資料庫版本（替換此模組即可，介面不變）
"""
from __future__ import annotations
import io
import csv
from typing import Optional
import pandas as pd

# ── 全域記憶體儲存 ─────────────────────────────────────────
_portfolio: list = []


# ── 讀取 ──────────────────────────────────────────────────

def get_all() -> list:
    return list(_portfolio)


def get_by_id(stock_id: str) -> Optional[dict]:
    stock_id = str(stock_id).zfill(4)
    for item in _portfolio:
        if item["stock_id"] == stock_id:
            return dict(item)
    return None


# ── 新增 / 更新 ───────────────────────────────────────────

def upsert(stock_id: str, name: str, shares: float, cost_price: float):
    """新增或更新單筆持股"""
    stock_id = str(stock_id).zfill(4)
    for item in _portfolio:
        if item["stock_id"] == stock_id:
            item["name"]       = name
            item["shares"]     = float(shares)
            item["cost_price"] = float(cost_price)
            return
    _portfolio.append({
        "stock_id":   stock_id,
        "name":       name,
        "shares":     float(shares),
        "cost_price": float(cost_price),
    })


# ── 刪除 ──────────────────────────────────────────────────

def remove(stock_id: str):
    global _portfolio
    stock_id = str(stock_id).zfill(4)
    _portfolio = [p for p in _portfolio if p["stock_id"] != stock_id]


def clear():
    global _portfolio
    _portfolio = []


# ── CSV 匯入 ──────────────────────────────────────────────

def import_csv_text(csv_text: str) -> tuple[int, str]:
    """
    從 CSV 文字匯入（覆蓋現有資料）
    回傳 (匯入筆數, 錯誤訊息或空字串)
    """
    try:
        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        required = {"stock_id", "name", "shares", "cost_price"}
        if not required.issubset(set(reader.fieldnames or [])):
            return 0, f"CSV 欄位錯誤，需要包含：{', '.join(required)}"

        rows = []
        for i, row in enumerate(reader, 1):
            try:
                rows.append({
                    "stock_id":   str(row["stock_id"]).zfill(4),
                    "name":       row["name"].strip(),
                    "shares":     float(row["shares"]),
                    "cost_price": float(row["cost_price"]),
                })
            except ValueError as e:
                return 0, f"第 {i} 行資料錯誤：{e}"

        if not rows:
            return 0, "CSV 沒有資料"

        global _portfolio
        _portfolio = rows
        return len(rows), ""

    except Exception as e:
        return 0, f"CSV 解析失敗：{e}"


# ── CSV 匯出 ──────────────────────────────────────────────

def export_csv_text() -> str:
    """匯出為 CSV 文字"""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["stock_id", "name", "shares", "cost_price"])
    writer.writeheader()
    writer.writerows(_portfolio)
    return output.getvalue()


# ── 轉 DataFrame（供批次分析使用） ───────────────────────────

def to_dataframe() -> pd.DataFrame:
    if not _portfolio:
        return pd.DataFrame(columns=["stock_id", "name", "shares", "cost_price"])
    return pd.DataFrame(_portfolio)
