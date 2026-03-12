"""
config.py
設定檔 - 所有敏感資訊從環境變數讀取
本機開發：在 shell 執行 export KEY=value，或建立 .env 後 source .env
Zeabur 部署：在 Environment Variables UI 填入對應值
"""
import os

# ── AI 模型設定 ────────────────────────────────────────────
# AI_PROVIDER: "claude" 或 "gemini"（預設 claude）
AI_PROVIDER = os.environ.get("AI_PROVIDER", "claude").lower()

# ── API 金鑰 ──────────────────────────────────────────────
FINMIND_TOKEN      = os.environ.get("FINMIND_TOKEN", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
FINLAB_API_TOKEN   = os.environ.get("FINLAB_API_TOKEN", "")

# ── Telegram 通知 ─────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")

# ── 技術指標參數 ──────────────────────────────────────────
MACD_FAST   = 6
MACD_SLOW   = 13
MACD_SIGNAL = 9
RSI_PERIOD  = 14
DATA_DAYS   = 120

# ── 基本面篩選門檻 ──────────────────────────────────────────
# 營收 YoY 門檻（%）：回測最佳值 0%（任何正成長）
# 回測期間 2021~2026，YoY>0% 年化+20.4% / 勝率41.6% / MDD 14.6%
# 原先 20% 門檻：年化+19.9% / 勝率35.7% / MDD 21.1%
PROFITABILITY_REV_YOY_THRESHOLD = 0

# ── 報告輸出（本機用，Zeabur 不會寫磁碟） ──────────────────
REPORTS_DIR    = "reports"
PORTFOLIO_FILE = "portfolio.csv"
