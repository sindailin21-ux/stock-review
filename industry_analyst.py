"""
industry_analyst.py
AI 產業分析報告生成模組 — 使用 Claude API。

功能：
1. get_industry_report(ticker, name, industry) → 產業報告（含快取）
2. get_monthly_revenue(ticker) → 月營收查詢（FinMind API）
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

from config import ANTHROPIC_API_KEY, FINMIND_TOKEN

# ═══════════════════════════════════════════════════════════
# 快取設定
# ═══════════════════════════════════════════════════════════

CACHE_DIR = Path("cache/industry_reports")
CACHE_DAYS = 7


def _get_cached(ticker: str):
    """讀取快取（7 天內有效）"""
    path = CACHE_DIR / f"{ticker}.json"
    if not path.exists():
        return None
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > timedelta(days=CACHE_DAYS):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(ticker: str, report: dict):
    """寫入快取"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{ticker}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ═══════════════════════════════════════════════════════════
# 月營收查詢（FinMind API）
# ═══════════════════════════════════════════════════════════

def get_monthly_revenue(ticker: str) -> dict:
    """
    從 FinMind 抓取近 6 個月月營收。
    回傳 {
        "source": "finmind" | "none",
        "records": [{"year": 2026, "month": 2, "revenue": 1261313000}, ...],
        "formatted": "2025/09: 12.09 億\n2025/10: ...",
        "note": "..."
    }
    """
    result = {
        "source": "none",
        "records": [],
        "formatted": "",
        "raw_html": "",
        "note": "月營收資料查詢失敗",
    }

    try:
        start = (datetime.now() - timedelta(days=240)).strftime("%Y-%m-%d")
        url = "https://api.finmindtrade.com/api/v4/data"
        params = {
            "dataset": "TaiwanStockMonthRevenue",
            "data_id": ticker,
            "start_date": start,
            "token": FINMIND_TOKEN,
        }
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()

        if not data.get("data"):
            result["note"] = f"FinMind 無營收資料：{data.get('msg', '')}"
            return result

        rows = data["data"][-6:]  # 取近 6 個月
        records = []
        lines = []
        for i, row in enumerate(rows):
            rev = row["revenue"]
            y = row["revenue_year"]
            m = row["revenue_month"]
            records.append({"year": y, "month": m, "revenue": rev})

            mom = ""
            if i > 0:
                prev_rev = rows[i - 1]["revenue"]
                if prev_rev > 0:
                    mom = f", 月增率: {(rev / prev_rev - 1) * 100:+.1f}%"
            lines.append(f"{y}/{m:02d}: {rev / 1e8:.2f} 億{mom}")

        # 生成原始數據 HTML 表格
        table_rows = ""
        for i, row in enumerate(rows):
            rev = row["revenue"]
            y = row["revenue_year"]
            m = row["revenue_month"]
            mom_str = "—"
            if i > 0:
                prev_rev = rows[i - 1]["revenue"]
                if prev_rev > 0:
                    mom_val = (rev / prev_rev - 1) * 100
                    color = "#4ecdc4" if mom_val >= 0 else "#ff6b6b"
                    mom_str = f'<span style="color:{color}">{mom_val:+.1f}%</span>'
            table_rows += f"<tr><td>{y}/{m:02d}</td><td style='text-align:right'>{rev / 1e8:.2f} 億</td><td style='text-align:right'>{mom_str}</td></tr>"

        raw_html = f"""<table style="width:100%;border-collapse:collapse;margin-bottom:12px">
<thead><tr style="border-bottom:1px solid #555"><th style="text-align:left">月份</th><th style="text-align:right">營收</th><th style="text-align:right">月增率</th></tr></thead>
<tbody>{table_rows}</tbody></table>"""

        result["source"] = "finmind"
        result["records"] = records
        result["formatted"] = "\n".join(lines)
        result["raw_html"] = raw_html
        result["note"] = f"來源：FinMind（近 {len(records)} 個月）"
        return result

    except Exception as e:
        print(f"  ⚠️ 月營收查詢失敗：{e}")
        return result


# ═══════════════════════════════════════════════════════════
# Gemini 產業報告生成
# ═══════════════════════════════════════════════════════════

def _build_industry_prompt(ticker: str, name: str, industry: str,
                           revenue_info: dict) -> str:
    """建構產業分析 Prompt"""

    # 月營收資料
    rev_section = ""
    if revenue_info.get("formatted"):
        rev_section = f"""
以下是該公司近 6 個月的月營收數據：
{revenue_info['formatted']}

請根據以上真實數據分析營收趨勢，包括：
- 營收趨勢判斷（成長/衰退/持平，是否有季節性）
- 關鍵觀察（是否突破新高？連續成長或衰退？轉折訊號？）
- 下月營收觀察重點
"""
    else:
        rev_section = f"""
月營收資料暫無法取得。
請根據你的知識，說明該公司近期營收概況與觀察重點。
"""

    return f"""你是一位專業的台股產業研究員，請為以下個股撰寫一份產業地位與展望分析報告。

## 個股資訊
- 股票代號：{ticker}
- 公司名稱：{name}
- 所屬產業：{industry or '（請自行判斷）'}

## 報告要求

請輸出嚴格符合以下 JSON 格式，每個欄位的值是 **HTML 格式的繁體中文內容**。
不要加任何其他文字、不要加 markdown 標記（如 ```json）。

**重要**：直接輸出 JSON，不要有任何前綴或後綴文字。

{{
    "positioning": "（產業定位與核心業務的 HTML）",
    "growth": "（產業展望與利多題材的 HTML）",
    "peers": "（競爭對手對照表的 HTML，必須包含 <table> 表格）",
    "revenue": "（月營收分析的 HTML）"
}}

### 各區塊內容要求：

**positioning（產業定位與核心業務）：**
- 核心身份（所屬集團、產業鏈角色）
- 主力產品線 2-3 項（含技術亮點與應用場景）
- 使用 <ul><li> 列點格式

**growth（產業展望與利多題材）：**
- 列出 2-3 個成長動能（如 AI 趨勢、新規格導入、去庫存循環、新應用場景等）
- 每個題材用 <p><b>標題</b>：說明</p> 格式
- 內容要具體、有深度，不要泛泛而談

**peers（同類型競爭對手對照表）：**
- 必須包含一個 HTML <table>，欄位為：競爭對手、代號、核心差異
- 列出 2-4 家同類型台股上市櫃公司
- 核心差異要具體說明各家的強項與差異點

**revenue（月營收分析）：**
{rev_section}
- 使用 <ul><li> 列點格式

所有內容請用繁體中文撰寫，HTML 標籤要正確閉合。"""


def _call_ai(prompt: str) -> dict:
    """呼叫 Claude API 生成產業報告"""
    import anthropic

    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=90.0,
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system="你是一位專業的台股產業研究員。請務必以繁體中文回答，輸出嚴格符合 JSON 格式，不要加任何其他文字或 markdown 標記。",
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_response(response.content[0].text)


def _parse_response(raw: str) -> dict:
    """解析 AI 回傳的 JSON，增強容錯"""
    raw = raw.strip()

    # 移除 markdown 包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        start = 1
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        raw = "\n".join(lines[start:end]).strip()

    # 嘗試直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 嘗試從回傳文字中擷取第一個 JSON 物件
    import re
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"無法從 AI 回應中解析 JSON，原始回應前 200 字：{raw[:200]}")


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def get_industry_report(ticker: str, name: str, industry: str = "") -> dict:
    """
    取得 AI 產業分析報告（含 7 天快取）。

    回傳 {
        "ticker": "4952",
        "name": "凌通",
        "generated_at": "2026-03-06",
        "cached": True/False,
        "sections": {
            "positioning": "...(HTML)",
            "growth": "...(HTML)",
            "peers": "...(HTML，含 <table>)",
            "conference": "...(HTML)",
        }
    }
    """
    # 1. 檢查快取
    cached = _get_cached(ticker)
    if cached:
        print(f"  📋 產業報告快取命中：{ticker} {name}")
        cached["cached"] = True
        return cached

    # 2. 查詢月營收
    print(f"  🔍 查詢月營收：{ticker}...")
    revenue_info = get_monthly_revenue(ticker)

    # 3. 呼叫 Claude
    if not ANTHROPIC_API_KEY:
        return _fallback_report(ticker, name, industry, revenue_info)

    try:
        print(f"  🤖 Claude 產業報告生成中：{ticker} {name}...")
        prompt = _build_industry_prompt(ticker, name, industry, revenue_info)
        sections = _call_ai(prompt)

        report = {
            "ticker": ticker,
            "name": name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "cached": False,
            "sections": {
                "positioning": sections.get("positioning", ""),
                "growth": sections.get("growth", ""),
                "peers": sections.get("peers", ""),
                "revenue_raw": revenue_info.get("raw_html", ""),
                "revenue": sections.get("revenue", ""),
            },
        }

        # 4. 寫入快取
        _save_cache(ticker, report)
        print(f"  ✅ 產業報告完成：{ticker} {name}")
        return report

    except Exception as e:
        err_msg = str(e)
        print(f"  ⚠️ Claude 產業報告失敗：{err_msg}")
        report = _fallback_report(ticker, name, industry, revenue_info)
        # 辨識常見錯誤，給使用者明確提示
        if "credit balance is too low" in err_msg:
            report["error_hint"] = "⚠️ Anthropic API 餘額不足，請至 console.anthropic.com/settings/billing 儲值"
        elif "authentication_error" in err_msg or "invalid x-api-key" in err_msg:
            report["error_hint"] = "⚠️ Anthropic API Key 無效或已過期，請至 console.anthropic.com/settings/keys 重新產生"
        elif "rate_limit" in err_msg:
            report["error_hint"] = "⚠️ Anthropic API 請求頻率超限，請稍後再試"
        else:
            report["error_hint"] = f"Claude AI 暫時無法使用：{err_msg[:100]}"
        return report


def _fallback_report(ticker: str, name: str, industry: str,
                     revenue_info: dict) -> dict:
    """無 API Key 或 AI 失敗時的備用報告"""
    rev_html = "<p>月營收資料查詢中，請稍後再試。</p>"
    if revenue_info.get("records"):
        items = "".join(
            f"<li>{r['year']}/{r['month']:02d}：{r['revenue'] / 1e8:.2f} 億</li>"
            for r in revenue_info["records"]
        )
        rev_html = f"<ul>{items}</ul><p>（AI 營收分析需要 ANTHROPIC_API_KEY）</p>"

    return {
        "ticker": ticker,
        "name": name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cached": False,
        "sections": {
            "positioning": f"<p>{name}（{ticker}）屬於{industry or '待查'}產業。詳細產業定位分析需要 AI API 支援。</p>",
            "growth": "<p>產業展望分析需要 AI API 支援，請設定 ANTHROPIC_API_KEY。</p>",
            "peers": "<p>競爭對手分析需要 AI API 支援。</p>",
            "revenue_raw": revenue_info.get("raw_html", ""),
            "revenue": rev_html,
        },
    }
