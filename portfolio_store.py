"""
portfolio_store.py
JSON 檔案型持股清單管理

資料格式：
  [{"stock_id": "2330", "name": "台積電", "shares": 1000, "cost_price": 850.0}, ...]

儲存位置：portfolio_data.json（與 app.py 同目錄）
"""
from __future__ import annotations
import io
import csv
import json
import fcntl
from pathlib import Path
from typing import Optional
import pandas as pd

# ── JSON 檔案路徑 ─────────────────────────────────────────
_DATA_FILE = Path(__file__).parent / "portfolio_data.json"


def _read_file() -> list:
    """從 JSON 檔案讀取持股"""
    if not _DATA_FILE.exists():
        return []
    try:
        with open(_DATA_FILE, "r", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
            return data if isinstance(data, list) else []
    except (json.JSONDecodeError, Exception):
        return []


def _write_file(data: list):
    """寫入 JSON 檔案（含 file lock）"""
    with open(_DATA_FILE, "w", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        json.dump(data, f, ensure_ascii=False, indent=2)
        fcntl.flock(f, fcntl.LOCK_UN)


# ── 讀取 ──────────────────────────────────────────────────

def get_all() -> list:
    return _read_file()


def get_by_id(stock_id: str) -> Optional[dict]:
    stock_id = str(stock_id).zfill(4)
    for item in _read_file():
        if item["stock_id"] == stock_id:
            return dict(item)
    return None


# ── 新增 / 更新 ───────────────────────────────────────────

def upsert(stock_id: str, name: str, shares: float, cost_price: float):
    """新增或更新單筆持股"""
    stock_id = str(stock_id).zfill(4)
    portfolio = _read_file()
    for item in portfolio:
        if item["stock_id"] == stock_id:
            item["name"]       = name
            item["shares"]     = float(shares)
            item["cost_price"] = float(cost_price)
            _write_file(portfolio)
            return
    portfolio.append({
        "stock_id":   stock_id,
        "name":       name,
        "shares":     float(shares),
        "cost_price": float(cost_price),
    })
    _write_file(portfolio)


# ── 刪除 ──────────────────────────────────────────────────

def remove(stock_id: str):
    stock_id = str(stock_id).zfill(4)
    portfolio = [p for p in _read_file() if p["stock_id"] != stock_id]
    _write_file(portfolio)


def clear():
    _write_file([])


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

        _write_file(rows)
        return len(rows), ""

    except Exception as e:
        return 0, f"CSV 解析失敗：{e}"


# ── CSV 匯出 ──────────────────────────────────────────────

def export_csv_text() -> str:
    """匯出為 CSV 文字"""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["stock_id", "name", "shares", "cost_price"])
    writer.writeheader()
    writer.writerows(_read_file())
    return output.getvalue()


# ── 轉 DataFrame（供批次分析使用） ───────────────────────────

def to_dataframe() -> pd.DataFrame:
    portfolio = _read_file()
    if not portfolio:
        return pd.DataFrame(columns=["stock_id", "name", "shares", "cost_price"])
    return pd.DataFrame(portfolio)
