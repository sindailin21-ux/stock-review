"""
ai_analyzer.py
呼叫 AI API（Claude 或 Gemini）針對每檔股票產生覆盤觀點與明日操作建議
支援透過 config.py 的 AI_PROVIDER 切換：'claude' / 'gemini'
若無 API Key 則自動使用規則式備用分析
"""

import json
from config import AI_PROVIDER, ANTHROPIC_API_KEY, GEMINI_API_KEY


# ═══════════════════════════════════════════════════════════
# 共用 Prompt 建構
# ═══════════════════════════════════════════════════════════

def _build_prompt(stock_id, stock_name, portfolio_info, technical_signals,
                  fundamental_summary, triggered_strategies=None):
    """建構分析 prompt（Claude / Gemini 共用）"""
    cost_price = portfolio_info.get("cost_price", 0)
    shares     = portfolio_info.get("shares", 0)
    current    = technical_signals.get("close", 0)
    profit_pct = ((current - cost_price) / cost_price * 100) if cost_price else 0

    # 策略感知區塊
    strategy_section = """
## 操作風格指引
你是一位偏多操作的波段交易者，風格特點：
- **趨勢為王**：只要均線多頭排列（MA5 > MA10 > MA20）且量價配合，就應該「持有」或「加碼」，不要輕易看空。
- **回測是機會**：股價回測 MA10 或 MA20 且量縮，是最佳的加碼時機，不是減碼理由。
- **尊重多頭慣性**：在多頭趨勢中，RSI 70 以上是「強勢」不是「過熱」，不要因為 RSI 高就建議減碼。
- **減碼 / 停損的條件要明確**：只有在「跌破關鍵均線 + 放量下殺」才建議減碼，不能僅因短線漲多就看空。
- 停損價位通常設在 MA20 或前波低點，但不要動不動就叫人停損。

## 操作建議判定原則
- **加碼**：回測支撐不破 + 量縮洗盤完成 + 法人仍站買方
- **持有**：多頭趨勢完整，沒有明確轉弱訊號
- **觀望**：訊號混沌，多空不明
- **減碼**：跌破 MA10 且量增，或乖離過大（>15%）
- **停損**：跌破 MA20 且 MA20 斜率轉下，趨勢確認反轉
"""

    # 若有明確觸發策略
    strat_info = ""
    if triggered_strategies:
        strat_labels = {
            "A": "均線糾結起漲",
            "B": "多頭續強",
            "C": "W底反轉",
            "D": "籌碼同步",
            "E": "池A：回測支撐",
            "F": "池B：動能確認",
        }
        strat_list = ", ".join(f"{s}({strat_labels.get(s, s)})" for s in triggered_strategies)
        strat_info = f"""
## 本股觸發策略
觸發策略：{strat_list}
請特別針對觸發的策略模式，給出對應的操作建議。
- 若觸發 E（回測支撐），代表股價在多頭趨勢中回測 MA20 且量縮，這是低風險買點，應建議「加碼」。
- 若觸發 F（動能確認），代表四線多排 + 帶量突破，多頭結構完美，應建議「持有」或「加碼」。
- 若觸發 A（均線糾結起漲），代表盤整後準備發動，應建議「加碼」或「觀望等突破」。
"""

    return f"""你是一位專業的台股波段交易分析師，正在進行盤後覆盤分析。
請根據以下資料，對 {stock_id} {stock_name} 進行分析，並給出明日操作建議。

## 持股資訊
- 持有股數：{shares} 股
- 持有成本：{cost_price} 元
- 今日收盤：{current} 元
- 浮動損益：{round(profit_pct, 2)}%

## 技術面資料
{json.dumps(technical_signals, ensure_ascii=False, indent=2)}

## 籌碼面資料
{json.dumps(fundamental_summary.get('institutional', {}), ensure_ascii=False)}

## 基本面資料
- 月營收：{json.dumps(fundamental_summary.get('revenue', {}), ensure_ascii=False)}
- EPS：{json.dumps(fundamental_summary.get('eps', {}), ensure_ascii=False)}
{strategy_section}{strat_info}
請以繁體中文回答，每個欄位都要寫得詳細具體（每欄至少 30 字以上），輸出嚴格符合以下 JSON 格式，不要加任何其他文字：

{{
    "summary": "一句話描述今日股價表現與市場意義（至少 30 字）",
    "technical_view": "技術面分析：詳述均線排列（MA5/MA20/MA60 相對位置）、MACD 柱狀體方向與金叉死叉狀態、RSI 超買超賣判讀、量價配合情況，至少 3-4 句",
    "chip_view": "籌碼面分析：外資今日買賣超張數與近 5 日趨勢、投信買賣超張數與近 5 日趨勢、自營商動向，至少 3 句，若資料不足請說明",
    "fundamental_view": "基本面觀點：月營收年增率與月增率趨勢、近幾季 EPS 表現、產業展望，至少 2-3 句，若資料不足請說明",
    "action": "持有 或 加碼 或 減碼 或 觀望 或 停損（多頭趨勢中偏向持有或加碼）",
    "action_reason": "明日操作建議與理由：結合技術面、籌碼面給出具體操作建議，包含建議的進出場價位，至少 3 句",
    "stop_loss": "建議停損價位與依據：明確寫出停損價格數字，並說明為何選擇這個價位（例如跌破某均線或前低），至少 2 句",
    "reversal_trigger": "反轉確認條件：明確寫出需要觀察的價位與量能條件，例如站回某均線且量能放大到多少倍，至少 2 句",
    "risk": "最主要的風險提示：具體說明可能的風險因素與應對方式，至少 2 句",
    "stop_loss_price": "純數字，停損價格（例如 820），只寫數字不要單位",
    "reversal_bull": "多方反轉：股價站上 XXX 元（必須寫具體價格數字）& 成交量放大 X.X 倍（必須寫具體倍數），一句話",
    "reversal_bear": "空方確認：股價跌破 XXX 元（必須寫具體價格數字）& 量縮至 X.X 倍以下（必須寫具體倍數），一句話"
}}"""


def _parse_ai_response(raw: str) -> dict:
    """解析 AI 回傳的 JSON（去除 markdown 包裹）"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


# ═══════════════════════════════════════════════════════════
# Claude 後端
# ═══════════════════════════════════════════════════════════

def _analyze_with_claude(prompt: str) -> dict:
    """使用 Anthropic Claude API 分析"""
    import anthropic
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=60.0,  # 60 秒 timeout，避免 hang 住
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system="你是一位專業的台股波段交易分析師。請務必詳細回答每個欄位，每個欄位至少寫 30 字以上，要有具體的數據、價位和理由。不要寫簡短的一句話。",
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_ai_response(response.content[0].text)


# ═══════════════════════════════════════════════════════════
# Gemini 後端
# ═══════════════════════════════════════════════════════════

def _analyze_with_gemini(prompt: str) -> dict:
    """使用 Google Gemini API 分析（新版 google-genai SDK）"""
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "temperature": 0.3,
            "max_output_tokens": 4096,
        },
    )
    return _parse_ai_response(response.text)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════

def analyze_stock(stock_id, stock_name, portfolio_info, technical_signals,
                  fundamental_summary, triggered_strategies=None):
    """
    分析單一股票。
    依 config.AI_PROVIDER 決定使用 Claude 或 Gemini，
    無 API Key 或 AI 失敗時自動退回規則式分析。
    triggered_strategies: list[str] 觸發的策略代碼（如 ["B", "E"]），可選。
    """
    provider = AI_PROVIDER  # 'claude' or 'gemini'

    # 判斷是否有可用的 API Key
    if provider == "gemini" and GEMINI_API_KEY:
        api_label = "Gemini"
        call_fn = _analyze_with_gemini
    elif provider == "claude" and ANTHROPIC_API_KEY:
        api_label = "Claude"
        call_fn = _analyze_with_claude
    elif ANTHROPIC_API_KEY:
        # 未指定或指定的 provider 沒有 key → 退回到有 key 的那個
        api_label = "Claude"
        call_fn = _analyze_with_claude
    elif GEMINI_API_KEY:
        api_label = "Gemini"
        call_fn = _analyze_with_gemini
    else:
        print(f"  📋 規則式分析中...（未設定任何 AI API Key）")
        return _fallback_analysis(technical_signals, portfolio_info, fundamental_summary)

    try:
        print(f"  🤖 {api_label} 分析中...")
        prompt = _build_prompt(stock_id, stock_name, portfolio_info,
                               technical_signals, fundamental_summary,
                               triggered_strategies)
        return call_fn(prompt)
    except Exception as e:
        print(f"  ⚠️  {api_label} 分析失敗（{e}），改用規則式分析")
        return _fallback_analysis(technical_signals, portfolio_info, fundamental_summary)


# ═══════════════════════════════════════════════════════════
# 規則式備用分析
# ═══════════════════════════════════════════════════════════

def _fallback_analysis(signals, portfolio_info=None, fundamental_summary=None):
    """規則式備用分析"""
    rsi         = signals.get("rsi", 50) or 50
    macd_status = signals.get("macd_status", "")
    macd_hist   = signals.get("macd_hist", 0) or 0
    change_pct  = signals.get("change_pct", 0) or 0
    vol_ratio   = signals.get("vol_ratio", 1) or 1
    rsi_status  = signals.get("rsi_status", "中性")
    ma_align    = signals.get("ma_alignment", "")
    close       = signals.get("close", 0)
    cost        = (portfolio_info or {}).get("cost_price", 0)
    profit_pct  = ((close - cost) / cost * 100) if cost else 0

    # 操作建議邏輯
    if rsi > 75:
        action = "減碼"
        reason = f"RSI {rsi:.1f} 進入超買區，技術面過熱，建議逢高減碼 1/3 部位，設定停利點。"
    elif rsi < 28:
        action = "加碼"
        reason = f"RSI {rsi:.1f} 進入超賣區，跌幅已深，可考慮分批承接，設定停損。"
    elif profit_pct < -8:
        action = "停損"
        reason = f"持股損益 {profit_pct:.1f}%，虧損幅度偏大，建議重新評估停損。"
    elif "多頭" in macd_status and macd_hist > 0 and change_pct > 0:
        action = "持有"
        reason = f"MACD {macd_status}，今日收漲 {change_pct:.2f}%，多頭趨勢延續，續抱觀察。"
    elif "空頭" in macd_status and macd_hist < 0 and change_pct < 0:
        action = "觀望"
        reason = f"MACD {macd_status}，空頭格局未改變，建議暫時觀望，等待訊號反轉。"
    else:
        action = "持有"
        reason = f"技術指標中性，持股損益 {profit_pct:+.1f}%，維持現有部位，持續觀察。"

    # 技術面描述
    tech = f"均線呈{ma_align}，RSI {rsi:.1f}（{rsi_status}），MACD {macd_status}（柱 {macd_hist:+.3f}）。"
    if vol_ratio >= 1.5:
        tech += f" 量能放大至均量 {vol_ratio:.1f} 倍，需留意方向確認。"

    # 籌碼面
    inst = (fundamental_summary or {}).get("institutional", {})
    if inst:
        parts = []
        for label, key in [("外資", "外資_今日(張)"), ("投信", "投信_今日(張)"), ("自營商", "自營商_今日(張)")]:
            val = inst.get(key)
            if val is not None:
                direction = "買超" if val > 0 else "賣超"
                parts.append(f"{label}{direction} {abs(val):,} 張")
        if parts:
            chip = "，".join(parts) + "。"
            five_d = []
            for label, key in [("外資", "外資_5日累計(張)"), ("投信", "投信_5日累計(張)")]:
                val = inst.get(key)
                if val is not None:
                    five_d.append(f"{label}五日累計{'買超' if val > 0 else '賣超'} {abs(val):,} 張")
            if five_d:
                chip += " " + "、".join(five_d) + "。"
        else:
            chip = "籌碼資料整理中。"
    else:
        chip = "籌碼資料整理中，請參考券商系統。"

    # 基本面
    rev = (fundamental_summary or {}).get("revenue", {})
    if rev:
        mom = rev.get("月增率(%)", None)
        yoy = rev.get("年增率(%)", None)
        fundamental = f"月營收月增率 {mom:+.1f}%，年增率 {yoy:+.1f}%。" if mom is not None and yoy is not None else "基本面資料整理中。"
    else:
        fundamental = "基本面資料整理中，請參考最新財報。"

    vol_note = f"量比 {vol_ratio:.1f}x，{'量能明顯放大。' if vol_ratio >= 1.5 else '量能平穩。'}"

    # 簡易停損與反轉點位（規則式）
    ma20 = signals.get("ma20")
    ma60 = signals.get("ma60")
    if ma20:
        stop_loss = f"建議跌破 MA20（{ma20} 元）即停損，代表短線支撐失守。"
        reversal  = f"若股價收復 MA20（{ma20} 元）且成交量放大至均量 1.5 倍以上，可考慮重新買回。"
    else:
        stop_loss = f"建議設定成本 -5% 為停損點（約 {round(cost * 0.95, 1) if cost else '-'} 元）。"
        reversal  = "若股價連續兩日收紅且量能回升，可觀察是否重新進場。"

    return {
        "summary": f"今日{'上漲' if change_pct > 0 else '下跌'} {abs(change_pct):.2f}%，{vol_note}",
        "technical_view": tech,
        "chip_view": chip,
        "fundamental_view": fundamental,
        "action": action,
        "action_reason": reason,
        "stop_loss": stop_loss,
        "reversal_trigger": reversal,
        "risk": "注意大盤整體走勢，若大盤跌破關鍵支撐須重新評估持股。",
    }
