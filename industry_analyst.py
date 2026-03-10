"""
industry_analyst.py
AI 產業分析報告生成模組 — 使用 Claude API。

功能：
1. get_industry_report(ticker, name, industry) → 產業報告（含快取）
2. get_investor_conference(ticker) → 法說會查詢（MOPS 公開資訊觀測站）
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
import pandas as pd

from config import ANTHROPIC_API_KEY

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
# 法說會查詢（MOPS 公開資訊觀測站）
# ═══════════════════════════════════════════════════════════

def get_investor_conference(ticker: str) -> dict:
    """
    嘗試從 MOPS 抓取法說會行事曆。
    回傳 {
        "source": "mops" | "ai",
        "events": [{"date": ..., "company": ..., "title": ...}],
        "note": "..."
    }
    """
    result = {
        "source": "ai",
        "events": [],
        "note": "由 AI 推估（MOPS 查詢失敗或無資料）",
    }

    try:
        # MOPS 法說會查詢頁面
        url = "https://mops.twse.com.tw/mops/web/t100sb02_q1"
        today = datetime.now()
        # 查詢近 30 天的法說會
        start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
        end_date = (today + timedelta(days=30)).strftime("%Y%m%d")

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://mops.twse.com.tw/mops/web/t100sb02",
        }

        payload = {
            "encodeURIComponent": "1",
            "step": "1",
            "firstin": "1",
            "off": "1",
            "TYPEK": "all",
            "co_id": ticker,
            "b_date": start_date,
            "e_date": end_date,
        }

        resp = requests.post(url, data=payload, headers=headers, timeout=15)
        resp.encoding = "utf-8"

        if resp.status_code == 200 and "<table" in resp.text.lower():
            try:
                tables = pd.read_html(resp.text)
                for tbl in tables:
                    if len(tbl) > 0 and len(tbl.columns) >= 3:
                        events = []
                        for _, row in tbl.iterrows():
                            vals = [str(v).strip() for v in row.values if str(v).strip() and str(v) != "nan"]
                            if len(vals) >= 2:
                                events.append({
                                    "date": vals[0] if vals else "",
                                    "company": vals[1] if len(vals) > 1 else "",
                                    "title": vals[2] if len(vals) > 2 else "",
                                })
                        if events:
                            result["source"] = "mops"
                            result["events"] = events[:5]  # 最多 5 筆
                            result["note"] = f"來源：公開資訊觀測站（共 {len(events)} 筆）"
                            return result
            except Exception:
                pass

        # 若無找到資料
        result["note"] = "近期無已排定之法說會行程（來源：MOPS）"
        return result

    except Exception as e:
        print(f"  ⚠️ MOPS 法說會查詢失敗：{e}")
        return result


# ═══════════════════════════════════════════════════════════
# Gemini 產業報告生成
# ═══════════════════════════════════════════════════════════

def _build_industry_prompt(ticker: str, name: str, industry: str,
                           conference_info: dict) -> str:
    """建構產業分析 Prompt"""

    # 法說會資料
    conf_section = ""
    if conference_info.get("events"):
        events_text = "\n".join(
            f"  - {e.get('date', '')} {e.get('company', '')} {e.get('title', '')}"
            for e in conference_info["events"]
        )
        conf_section = f"""
以下是從公開資訊觀測站查到的法說會資料：
{events_text}
請根據以上資料撰寫法說會資訊區塊。
"""
    else:
        conf_section = f"""
公開資訊觀測站查詢結果：{conference_info.get('note', '近期無法說會行程')}
請根據你的知識，說明該公司通常的法說會舉辦時間、頻率、觀察重點。
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
    "conference": "（法說會資訊的 HTML）"
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

**conference（法說會資訊）：**
{conf_section}
- 說明最新動態、關鍵觀察期、觀察重點
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
    """解析 Gemini 回傳的 JSON"""
    raw = raw.strip()
    # 移除 markdown 包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        # 找第一個 ``` 後的內容
        start = 1
        if lines[0].startswith("```json"):
            start = 1
        # 找最後一個 ```
        end = len(lines)
        for i in range(len(lines) - 1, 0, -1):
            if lines[i].strip() == "```":
                end = i
                break
        raw = "\n".join(lines[start:end])

    return json.loads(raw.strip())


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

    # 2. 查詢法說會
    print(f"  🔍 查詢法說會：{ticker}...")
    conference_info = get_investor_conference(ticker)

    # 3. 呼叫 Claude
    if not ANTHROPIC_API_KEY:
        return _fallback_report(ticker, name, industry, conference_info)

    try:
        print(f"  🤖 Claude 產業報告生成中：{ticker} {name}...")
        prompt = _build_industry_prompt(ticker, name, industry, conference_info)
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
                "conference": sections.get("conference", ""),
            },
        }

        # 4. 寫入快取
        _save_cache(ticker, report)
        print(f"  ✅ 產業報告完成：{ticker} {name}")
        return report

    except Exception as e:
        err_msg = str(e)
        print(f"  ⚠️ Claude 產業報告失敗：{err_msg}")
        report = _fallback_report(ticker, name, industry, conference_info)
        report["error_hint"] = f"Claude AI 暫時無法使用：{err_msg[:100]}"
        return report


def _fallback_report(ticker: str, name: str, industry: str,
                     conference_info: dict) -> dict:
    """無 API Key 或 AI 失敗時的備用報告"""
    conf_html = "<p>法說會資料查詢中，請參考公開資訊觀測站。</p>"
    if conference_info.get("events"):
        items = "".join(
            f"<li>{e.get('date', '')} — {e.get('title', '')}</li>"
            for e in conference_info["events"]
        )
        conf_html = f"<ul>{items}</ul>"

    return {
        "ticker": ticker,
        "name": name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cached": False,
        "sections": {
            "positioning": f"<p>{name}（{ticker}）屬於{industry or '待查'}產業。詳細產業定位分析需要 AI API 支援。</p>",
            "growth": "<p>產業展望分析需要 AI API 支援，請設定 ANTHROPIC_API_KEY。</p>",
            "peers": "<p>競爭對手分析需要 AI API 支援。</p>",
            "conference": conf_html,
        },
    }
