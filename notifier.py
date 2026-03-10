"""
notifier.py
發送 Telegram 通知，摘要今日覆盤結果
"""

import requests


def send_telegram(token, chat_id, message, parse_mode="HTML"):
    """發送 Telegram 訊息"""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  ⚠️  Telegram 通知失敗：{e}")
        return False


def build_summary_message(all_stock_data, report_date):
    """組合摘要訊息"""

    ACTION_EMOJI = {
        "加碼": "📈",
        "持有": "🔵",
        "觀望": "🟡",
        "減碼": "🟠",
        "停損": "🔴",
    }

    # 統計
    total   = len(all_stock_data)
    up      = sum(1 for s in all_stock_data if s["signals"].get("change_pct", 0) > 0)
    down    = sum(1 for s in all_stock_data if s["signals"].get("change_pct", 0) < 0)
    flat    = total - up - down

    total_profit = sum(
        (s["signals"].get("close", 0) - s["portfolio"].get("cost_price", 0))
        * s["portfolio"].get("shares", 0)
        for s in all_stock_data
    )
    profit_sign = "+" if total_profit >= 0 else ""
    profit_emoji = "🔺" if total_profit >= 0 else "🔻"

    # 表頭
    lines = [
        f"📊 <b>台股盤後覆盤｜{report_date}</b>",
        f"",
        f"持股 {total} 檔｜漲 {up} 跌 {down} 平 {flat}",
        f"{profit_emoji} 浮動損益：<b>{profit_sign}{total_profit:,.0f} 元</b>",
        f"",
        f"━━━━━━━━━━━━━━━━",
    ]

    # 排序：停損 > 減碼 > 加碼 > 觀望 > 持有
    order = {"停損": 0, "減碼": 1, "加碼": 2, "觀望": 3, "持有": 4}
    sorted_data = sorted(
        all_stock_data,
        key=lambda s: order.get(s["analysis"].get("action", "觀望"), 5)
    )

    for s in sorted_data:
        stock_id   = s["stock_id"]
        name       = s["name"]
        signals    = s["signals"]
        analysis   = s["analysis"]
        portfolio  = s["portfolio"]

        close      = signals.get("close", 0)
        change_pct = signals.get("change_pct", 0)
        cost       = portfolio.get("cost_price", 0)
        profit_pct = ((close - cost) / cost * 100) if cost else 0
        action     = analysis.get("action", "觀望")
        reason     = analysis.get("action_reason", "")
        stop_loss  = analysis.get("stop_loss", "")

        change_arrow = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "－"
        emoji = ACTION_EMOJI.get(action, "⚪")

        lines.append(f"{emoji} <b>{stock_id} {name}</b>")
        lines.append(f"   收盤 {close}（{change_arrow}{abs(change_pct):.2f}%）｜損益 {profit_pct:+.1f}%")
        lines.append(f"   建議：<b>{action}</b>　{reason[:40]}{'...' if len(reason) > 40 else ''}")
        if stop_loss:
            lines.append(f"   🛑 {stop_loss[:40]}{'...' if len(stop_loss) > 40 else ''}")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━")
    lines.append("📁 詳細報告請開啟 HTML 檔案查看")

    return "\n".join(lines)


def send_html_report(token, chat_id, filepath, report_date):
    """傳送 HTML 報告檔案給使用者"""
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    try:
        with open(filepath, "rb") as f:
            resp = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "caption": f"📁 {report_date} 完整覆盤報告\n下載後用瀏覽器開啟查看",
                },
                files={"document": (f"report_{report_date.replace('/', '-')}.html", f, "text/html")},
                timeout=30,
            )
            resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  ⚠️  HTML 報告傳送失敗：{e}")
        return False


def notify(all_stock_data, report_date, token, chat_id, html_filepath=None):
    """對外介面：組訊息並發送，可附上 HTML 報告"""
    if not token or not chat_id:
        print("  ⚠️  未設定 TELEGRAM_TOKEN 或 TELEGRAM_CHAT_ID，跳過通知")
        return

    print("  📲 發送 Telegram 通知中...")
    message = build_summary_message(all_stock_data, report_date)
    success = send_telegram(token, chat_id, message)
    if success:
        print("  ✅ Telegram 通知已發送")

    # 傳送 HTML 報告檔案
    if html_filepath:
        print("  📎 傳送 HTML 報告中...")
        success = send_html_report(token, chat_id, html_filepath, report_date)
        if success:
            print("  ✅ HTML 報告已傳送")
