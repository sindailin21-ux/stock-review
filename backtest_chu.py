#!/usr/bin/env python3
"""
backtest_chu.py
朱家泓（Master Chu）進場法 — FinLab 回測腳本（v2 優化版）

策略條件（全部滿足才進場）：
  1. 頭頭高底底高：近 40 日擺盪高低點呈上升型態
  2. 四線多排 + 正斜率：MA5 > MA10 > MA20 > MA60，且四線斜率皆 > 0
  3. MA20 扣抵值看多：收盤 > 20 日前收盤（MA20 自然上升）
  4. 攻擊量：收盤 > MA5 且量 > VMA5×1.5 且收紅K 且上影線<30% 且實體>1%
  5. 乖離率濾網：收盤與 MA5 乖離率 < 3%（避免追高）
  6. 大盤濾網：加權指數 > MA20（大盤多頭才做多）
  7. 基本面濾網：最近一季 EPS > 0 且 月營收 YoY > 0%

出場條件（任一觸發即出場）：
  - 收盤跌破 5 日線
  - 5 日線斜率由正轉平/負
  - 收盤 < 進場價 × (1 - stop_loss)（強制停損）

Usage:
    python backtest_chu.py
    python backtest_chu.py --years 3 --nstocks 15
    python backtest_chu.py --years 5 --stop-loss 0.05 --max-bias 0.03
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════
# 本地磁碟快取（節省 FinLab API 用量）
# ═══════════════════════════════════════════════════════════

CACHE_DIR = "cache/finlab_data"


def _load_data_with_cache(data_module):
    """
    載入收盤價 & 成交股數。
    先嘗試讀本地 pickle；沒有或過期則從 FinLab 下載並存檔。
    快取有效期 1 天。
    """
    import os, pickle
    from datetime import datetime, timedelta

    os.makedirs(CACHE_DIR, exist_ok=True)
    close_path = os.path.join(CACHE_DIR, "close.pkl")
    vol_path = os.path.join(CACHE_DIR, "volume.pkl")
    bench_path = os.path.join(CACHE_DIR, "benchmark.pkl")

    # 快取有效期：1 天
    def _is_fresh(path):
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) < timedelta(days=1)

    if _is_fresh(close_path) and _is_fresh(vol_path):
        print("📥 載入本地快取資料...")
        with open(close_path, "rb") as f:
            close = pickle.load(f)
        with open(vol_path, "rb") as f:
            volume = pickle.load(f)
        print(f"   ✅ 快取命中（免耗 FinLab 用量）")
    else:
        print("📥 從 FinLab 下載歷史資料...")
        close = data_module.get("price:收盤價")
        volume = data_module.get("price:成交股數")
        # 存入快取
        with open(close_path, "wb") as f:
            pickle.dump(close, f)
        with open(vol_path, "wb") as f:
            pickle.dump(volume, f)
        print("   💾 已存入本地快取")

    return close, volume


def _load_benchmark_with_cache(data_module):
    """載入大盤加權指數（含快取）"""
    import os, pickle
    from datetime import datetime, timedelta

    os.makedirs(CACHE_DIR, exist_ok=True)
    bench_path = os.path.join(CACHE_DIR, "benchmark.pkl")

    def _is_fresh(path):
        if not os.path.exists(path):
            return False
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        return (datetime.now() - mtime) < timedelta(days=1)

    if _is_fresh(bench_path):
        with open(bench_path, "rb") as f:
            return pickle.load(f)
    else:
        benchmark = data_module.get("benchmark_return:發行量加權股價報酬指數")
        with open(bench_path, "wb") as f:
            pickle.dump(benchmark, f)
        return benchmark


# ═══════════════════════════════════════════════════════════
# 命令列參數
# ═══════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="朱家泓進場法 FinLab 回測",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--years", type=int, default=5,
                        help="回測年數 (default: 5)")
    parser.add_argument("--nstocks", type=int, default=10,
                        help="最大同時持股數 (default: 10)")
    parser.add_argument("--stop-loss", type=float, default=0.08,
                        help="停損比例 (default: 0.08 = 8%%)")
    parser.add_argument("--take-profit", type=float, default=None,
                        help="停利比例 (default: None，由 MA5 出場)")
    parser.add_argument("--lookback", type=int, default=40,
                        help="Swing detection 回看天數 (default: 40)")
    parser.add_argument("--vol-mult", type=float, default=1.5,
                        help="攻擊量倍數 (default: 1.5)")
    parser.add_argument("--max-bias", type=float, default=0.03,
                        help="MA5 乖離率上限 (default: 0.03 = 3%%)")
    parser.add_argument("--no-market-filter", action="store_true",
                        help="停用大盤濾網")
    parser.add_argument("--no-fundamental", action="store_true",
                        help="停用基本面濾網（EPS>0 且 營收YoY>0）")
    parser.add_argument("--exit-ma", type=int, default=0,
                        help="出場均線：5=MA5, 10=MA10, 20=MA20, 0=動態(default)")
    parser.add_argument("--compare", action="store_true",
                        help="A/B/C 三組出場策略對比模式")
    parser.add_argument("--stocks", type=str, default=None,
                        help="自訂股票清單，逗號分隔 (例: 2330,2454,2317)")
    parser.add_argument("--watch", type=str, default=None,
                        help="關注清單：全市場選股但額外分析這些股票的交易 (例: 2330,2454)")
    parser.add_argument("--min-hold", type=int, default=3,
                        help="最低持有天數，緩衝期內不因技術面出場 (default: 3)")
    parser.add_argument("--atr-stop", action="store_true",
                        help="使用 ATR 動態停損（2×ATR14）取代固定比例停損")
    parser.add_argument("--relax-entry", action="store_true",
                        help="放寬進場：移除上影線/實體限制，量能降至 1.2x")
    parser.add_argument("--no-ma60", action="store_true",
                        help="移除 MA60 約束：3 線多排 (MA5>MA10>MA20) 取代 4 線")
    parser.add_argument("--trail-stop", type=float, default=None,
                        help="移動停利：持有期間最高價回落 N%% 出場 (例: 0.08=8%%)")
    parser.add_argument("--full-invest", action="store_true",
                        help="動態權重：持股平均分配 100%% 資金（消除現金閒置）")
    parser.add_argument("--optimize", action="store_true",
                        help="全方位優化模式：自動測試多組參數組合")
    parser.add_argument("--adx", action="store_true",
                        help="用 ADX(14) 趨勢強度取代擺盪偵測（最佳策略）")
    parser.add_argument("--adx-period", type=int, default=14,
                        help="ADX 計算天期（預設 14，可試 7/10/12）")
    parser.add_argument("--best", action="store_true",
                        help="一鍵套用最佳策略（ADX趨勢+放寬進場+RSI+停損6%%+停利15%%+持有5天）")
    parser.add_argument("--rev-yoy", type=float, default=0,
                        help="營收 YoY 門檻 (default: 0 = 任何正成長，0.2 = 20%%)")
    parser.add_argument("--dist-filter", action="store_true",
                        help="高檔出貨過濾：長黑K + 位階>70%% + 法人賣超 → 排除進場")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════
# 條件 1：頭頭高底底高（per-stock）
# ═══════════════════════════════════════════════════════════

def compute_swing_pattern(close_df: pd.DataFrame, lookback: int = 40) -> pd.DataFrame:
    """
    對每檔股票計算「頭頭高底底高」布林 DataFrame。

    方法：每檔呼叫 find_peaks 一次（全歷史），
    再對每個日期檢查前 lookback 日內是否有符合的擺盪型態。
    """
    result = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)

    for stock_id in tqdm(close_df.columns, desc="📊 擺盪型態偵測", ncols=80):
        series = close_df[stock_id].dropna()
        if len(series) < lookback + 5:
            continue

        arr = series.values.astype(float)
        avg_price = np.nanmean(arr)
        prominence = max(avg_price * 0.02, 0.5)

        # 全歷史找出所有擺盪高低點（只做一次）
        all_highs, _ = find_peaks(arr, distance=5, prominence=prominence)
        all_lows, _ = find_peaks(-arr, distance=5, prominence=prominence)

        if len(all_highs) < 2 or len(all_lows) < 2:
            continue

        # 轉為 numpy array 方便 searchsorted
        all_highs = np.array(all_highs)
        all_lows = np.array(all_lows)

        # 逐日檢查
        for i in range(lookback, len(arr)):
            window_start = i - lookback

            # 視窗內的高點
            h_mask = (all_highs >= window_start) & (all_highs <= i)
            h_in = all_highs[h_mask]

            l_mask = (all_lows >= window_start) & (all_lows <= i)
            l_in = all_lows[l_mask]

            if len(h_in) >= 2 and len(l_in) >= 2:
                hh = arr[h_in[-1]] > arr[h_in[-2]]  # 頭頭高
                hl = arr[l_in[-1]] > arr[l_in[-2]]  # 底底高
                if hh and hl:
                    result.loc[series.index[i], stock_id] = True

    return result


# ═══════════════════════════════════════════════════════════
# 進場 / 出場訊號
# ═══════════════════════════════════════════════════════════

def build_entry_signals(close, volume, ma5, ma10, ma20, ma60, vma5,
                        cond1_swing, vol_mult=1.5,
                        market_filter=None, max_bias=0.03,
                        open_=None, high=None, low=None,
                        fundamental_filter=None,
                        relax_entry=False,
                        no_ma60=False):
    """合成進場條件（含攻擊量 + 基本面 + 大盤濾網 + 乖離率）。

    relax_entry=True 時：
      - 量能只需 close > MA5 且 vol > VMA5×1.2（不要求紅K/上影線/實體）
      - 乖離率放寬至 5%
    no_ma60=True 時：
      - 3 線多排（MA5>MA10>MA20），不含 MA60
      - 斜率也只看 3 條線
    """

    # 條件 2：多排 + 正斜率
    if no_ma60:
        # 3 線多排（移除 MA60 約束 → 大幅增加進場訊號）
        cond_align = (ma5 > ma10) & (ma10 > ma20)
        cond_slope = (
            (ma5 > ma5.shift(3)) &
            (ma10 > ma10.shift(3)) &
            (ma20 > ma20.shift(3))
        )
    else:
        # 原始 4 線多排
        cond_align = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
        cond_slope = (
            (ma5 > ma5.shift(3)) &
            (ma10 > ma10.shift(3)) &
            (ma20 > ma20.shift(3)) &
            (ma60 > ma60.shift(3))
        )
    cond2 = cond_align & cond_slope

    # 條件 3：MA20 扣抵值看多
    cond3 = close > close.shift(20)

    # 條件 4：量能確認
    cond4a = close > ma5

    if relax_entry:
        # 放寬模式：量 > VMA5×1.2（不要求紅K，回測證明效果最好）
        # 跳空高開後小拉回的「黑K」也是有效多頭訊號
        cond4b = volume > vma5 * 1.2
        cond4 = cond4a & cond4b
    else:
        # 嚴格攻擊量模式
        cond4b = volume > vma5 * vol_mult
        if open_ is not None and high is not None and low is not None:
            k_range = high - low
            body = (close - open_).abs()
            upper_shadow = high - close.clip(upper=high)

            cond4c = close > open_                                          # 收紅 K
            cond4d = (upper_shadow <= k_range * 0.30) | (k_range == 0)     # 上影線 < 30%
            cond4e = (body / close) > 0.01                                  # 實體 > 1%

            cond4 = cond4a & cond4b & cond4c & cond4d & cond4e
        else:
            cond4 = cond4a & cond4b

    # 條件 5：乖離率（放寬模式 5%，嚴格模式用 max_bias）
    effective_bias = 0.05 if relax_entry else max_bias
    bias = (close - ma5) / ma5
    cond5 = bias < effective_bias

    # 合成
    entries = cond1_swing & cond2 & cond3 & cond4 & cond5

    # 條件 6：大盤濾網（加權指數 > MA20）
    if market_filter is not None:
        entries = entries & market_filter

    # 條件 7：基本面濾網（EPS > 0 且 營收 YoY > 0）
    if fundamental_filter is not None:
        entries = entries & fundamental_filter

    return entries.fillna(False)


def build_exit_signals(close, ma5, ma10, ma20, exit_ma: int = 0):
    """
    合成出場條件。

    exit_ma 模式：
      5  = 短線模式（A組）：跌破 MA5 出場
      10 = 慣性模式（B組）：跌破 MA10 出場
      20 = 波段模式（C組）：跌破 MA20 出場
      0  = 動態模式（D組）：飆股守MA10、穩健股守MA20 + MA5下彎跌破
      -1 = 寬鬆動態（E組）：只用MA10/MA20切換，不含MA5下彎
      -2 = 純移動停利（F組）：只靠停損/移動停利出場，技術面完全不管
    """
    ma5_slope = ma5 - ma5.shift(3)

    if exit_ma == 5:
        # A 組：短線 — 跌破 MA5
        exits = close < ma5
    elif exit_ma == 10:
        # B 組：慣性 — 跌破 MA10
        exits = close < ma10
    elif exit_ma == 20:
        # C 組：波段 — 跌破 MA20
        exits = close < ma20
    elif exit_ma == -1:
        # E 組：寬鬆動態 — 只用 MA10/MA20，不含 MA5 下彎
        bias_ma20 = (close - ma20) / ma20
        is_hot = bias_ma20 > 0.10
        exit_hot = is_hot & (close < ma10)
        exit_steady = (~is_hot) & (close < ma20)
        exits = exit_hot | exit_steady
    elif exit_ma == -2:
        # F 組：純移動停利 — 技術面永遠不出場，靠停損/移動停利
        exits = pd.DataFrame(False, index=close.index, columns=close.columns)
    else:
        # D 組：動態模式 — 飆股守 MA10、穩健股守 MA20 + MA5 下彎
        bias_ma20 = (close - ma20) / ma20
        is_hot = bias_ma20 > 0.10
        exit_hot = is_hot & (close < ma10)
        exit_steady = (~is_hot) & (close < ma20)
        # 通用：MA5 下彎 + 跌破 → 不論類型都跑
        exit_ma5_break = (ma5_slope <= 0) & (close < ma5)
        exits = exit_hot | exit_steady | exit_ma5_break

    return exits.fillna(False)


# ═══════════════════════════════════════════════════════════
# 報告輸出工具
# ═══════════════════════════════════════════════════════════

def _print_report_stats(report):
    """從 FinLab report 物件抓取績效指標"""
    try:
        # 使用 get_metrics() API
        metrics = report.get_metrics() if hasattr(report, "get_metrics") else {}
        if metrics:
            prof = metrics.get("profitability", {})
            risk = metrics.get("risk", {})
            ratio = metrics.get("ratio", {})

            annual_ret = float(prof.get("annualReturn", 0)) * 100
            max_dd = abs(float(risk.get("maxDrawdown", 0))) * 100
            sharpe = float(ratio.get("sharpeRatio", 0))

            print(f"  年化報酬率:   {annual_ret:+.2f}%")
            print(f"  夏普比率:     {sharpe:.2f}")
            print(f"  最大回撤:     {max_dd:.2f}%")
            return

        # 備用：從 creturn 計算
        if hasattr(report, "creturn"):
            equity = report.creturn
            if isinstance(equity, pd.Series) and len(equity) > 1:
                total_ret = equity.iloc[-1] / equity.iloc[0] - 1
                years = (equity.index[-1] - equity.index[0]).days / 365.25
                annual_ret = ((1 + total_ret) ** (1 / years) - 1) * 100 if years > 0 else 0
                rolling_max = equity.cummax()
                drawdown = (equity - rolling_max) / rolling_max
                max_dd = abs(drawdown.min()) * 100
                daily_ret = equity.pct_change().dropna()
                sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

                print(f"  年化報酬率:   {annual_ret:+.2f}%")
                print(f"  夏普比率:     {sharpe:.2f}")
                print(f"  最大回撤:     {max_dd:.2f}%")
                return

    except Exception as e:
        print(f"  ⚠️ FinLab 報告統計擷取異常：{e}")


def _print_trade_stats(position: pd.DataFrame, close: pd.DataFrame,
                       start_date: pd.Timestamp):
    """從持倉矩陣和收盤價計算交易統計"""
    print(f"  ──────────────────────────")

    try:
        # 偵測所有交易：找出每次持倉從 0 變非 0（進場）和從非 0 變 0（出場）
        trades = []
        held = position.astype(bool)

        for sid in held.columns:
            col = held[sid]
            in_trade = False
            entry_date = None

            for date in col.index:
                if col[date] and not in_trade:
                    # 進場
                    in_trade = True
                    entry_date = date
                elif not col[date] and in_trade:
                    # 出場
                    in_trade = False
                    exit_date = date
                    if sid in close.columns:
                        entry_price = close.loc[entry_date, sid] if entry_date in close.index else np.nan
                        exit_price = close.loc[exit_date, sid] if exit_date in close.index else np.nan
                        if not np.isnan(entry_price) and not np.isnan(exit_price) and entry_price > 0:
                            ret = (exit_price / entry_price) - 1
                            days = (exit_date - entry_date).days
                            trades.append({
                                "stock_id": sid,
                                "entry_date": entry_date,
                                "exit_date": exit_date,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "return": ret,
                                "days": days,
                            })

        total_trades = len(trades)
        print(f"  總交易次數:   {total_trades}")

        if total_trades == 0:
            print("=" * 56)
            return

        trades_df = pd.DataFrame(trades)
        win_trades = int((trades_df["return"] > 0).sum())
        lose_trades = total_trades - win_trades
        win_rate = win_trades / total_trades * 100
        avg_ret = trades_df["return"].mean() * 100
        avg_win = trades_df.loc[trades_df["return"] > 0, "return"].mean() * 100 if win_trades > 0 else 0
        avg_loss = trades_df.loc[trades_df["return"] <= 0, "return"].mean() * 100 if lose_trades > 0 else 0
        avg_days = trades_df["days"].mean()

        # 盈虧比
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        print(f"  勝率:         {win_rate:.1f}%（{win_trades}W / {lose_trades}L）")
        print(f"  平均報酬:     {avg_ret:+.2f}%")
        print(f"  平均獲利:     {avg_win:+.2f}%")
        print(f"  平均虧損:     {avg_loss:+.2f}%")
        print(f"  盈虧比:       {profit_loss_ratio:.2f}")
        print(f"  平均持有天數:  {avg_days:.1f}")
        print("=" * 56)

        # 顯示最近 10 筆交易
        print()
        print("  📝 最近 10 筆交易：")
        recent = trades_df.sort_values("exit_date").tail(10)
        for _, t in recent.iterrows():
            ret = t["return"] * 100
            icon = "🟢" if ret > 0 else "🔴"
            entry_d = t["entry_date"].strftime("%Y-%m-%d")
            exit_d = t["exit_date"].strftime("%Y-%m-%d")
            print(f"    {icon} {t['stock_id']}  {entry_d} → {exit_d}  {ret:+.1f}%  ({t['days']}天)")

    except Exception as e:
        print(f"  ⚠️ 交易統計計算異常：{e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════
# 持倉建構（手動 hold_until）
# ═══════════════════════════════════════════════════════════

def _build_position(entries: pd.DataFrame, exits: pd.DataFrame,
                    rank: pd.DataFrame, close: pd.DataFrame,
                    nstocks: int = 10, stop_loss: float = 0.05,
                    min_hold_days: int = 3,
                    atr_df: pd.DataFrame = None,
                    trail_stop: float = None,
                    full_invest: bool = False,
                    atr_trail_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    手動建構持倉矩陣（含停損 + 最低持有天數 + 移動停利）。

    邏輯：
    - 進場：entry=True 且未持有 → 開倉
    - 持有：直到 exit=True 或觸發停損/停利 → 平倉
    - 停損：固定比例 or ATR 動態停損（atr_df 不為 None 時）
    - 移動停利（trail_stop）：從持有期間最高價回落 trail_stop 比例出場
    - atr_trail_df：ATR 自適應移動停利（每檔每天不同的百分比）
    - 最低持有天數：進場後至少持有 min_hold_days 天才允許技術面出場
    - 同時最多持有 nstocks 檔，以 rank 排序選最強的
    - full_invest：動態權重，持倉平均分配 100%（消除現金閒置）
    """
    position = pd.DataFrame(0.0, index=entries.index, columns=entries.columns)
    # held: {stock_id: (entry_idx, entry_price, stop_price, highest_price)}
    held = {}

    stop_loss_count = 0
    trail_stop_count = 0
    use_atr = atr_df is not None
    use_trail = trail_stop is not None and trail_stop > 0
    use_atr_trail = atr_trail_df is not None
    total_invested_count = 0  # 用於統計平均持倉數

    dates = entries.index
    for i, date in enumerate(tqdm(dates, desc="📦 建構持倉", ncols=80)):
        # 1. 先處理出場：檢查已持有的股票是否觸發出場或停損
        to_exit = []
        for sid, (eidx, eprice, stop_price, highest) in list(held.items()):
            holding_days = i - eidx
            current_price = close.loc[date, sid] if sid in close.columns else np.nan

            if np.isnan(current_price) or eprice <= 0:
                continue

            # 更新最高價
            new_highest = max(highest, current_price)
            held[sid] = (eidx, eprice, stop_price, new_highest)

            # 停損出場：不受最低持有天數限制
            stop_exit = False
            if current_price < stop_price:
                stop_exit = True
                stop_loss_count += 1

            # 移動停利出場：持有期間最高價回落 trail_stop%
            trail_exit = False
            if holding_days >= min_hold_days:
                if use_trail and new_highest > eprice:
                    if current_price < new_highest * (1 - trail_stop):
                        trail_exit = True
                        trail_stop_count += 1
                elif use_atr_trail and new_highest > eprice:
                    # ATR 自適應停利：用當天該股的 ATR 百分比
                    atr_pct = 0.15  # fallback
                    if sid in atr_trail_df.columns and date in atr_trail_df.index:
                        val = atr_trail_df.loc[date, sid]
                        if not np.isnan(val):
                            atr_pct = val
                    if current_price < new_highest * (1 - atr_pct):
                        trail_exit = True
                        trail_stop_count += 1

            # 技術面出場：需滿足最低持有天數
            tech_exit = False
            if holding_days >= min_hold_days:
                tech_exit = exits.loc[date, sid] if sid in exits.columns else False

            if tech_exit or stop_exit or trail_exit:
                to_exit.append(sid)

        for sid in to_exit:
            del held[sid]

        # 2. 處理新進場：有空位才能進場
        available_slots = nstocks - len(held)
        if available_slots > 0:
            entry_today = entries.loc[date]
            candidates = entry_today[entry_today & ~entry_today.index.isin(held.keys())]

            if len(candidates) > 0:
                if date in rank.index:
                    candidate_ranks = rank.loc[date, candidates.index].dropna()
                    candidate_ranks = candidate_ranks.sort_values(ascending=False)
                    selected = candidate_ranks.head(available_slots).index
                else:
                    selected = candidates.head(available_slots).index

                for sid in selected:
                    eprice = close.loc[date, sid] if sid in close.columns else 0
                    if use_atr and sid in atr_df.columns and date in atr_df.index:
                        atr_val = atr_df.loc[date, sid]
                        if not np.isnan(atr_val) and atr_val > 0:
                            sp = eprice - 2.0 * atr_val
                        else:
                            sp = eprice * (1 - stop_loss)
                    else:
                        sp = eprice * (1 - stop_loss)
                    held[sid] = (i, eprice, sp, eprice)

        # 3. 設定權重
        n_held = len(held)
        if n_held > 0:
            if full_invest:
                # 動態權重：100% 平均分配給持有的股票
                weight = 1.0 / n_held
            else:
                # 固定權重：每檔 1/nstocks
                weight = 1.0 / nstocks
            for sid in held:
                position.loc[date, sid] = weight
            total_invested_count += n_held

    avg_held = total_invested_count / len(dates) if len(dates) > 0 else 0
    invest_pct = avg_held / nstocks * 100
    print(f"   平均持倉數：{avg_held:.1f}/{nstocks}（資金利用率 {invest_pct:.0f}%）")
    print(f"   停損觸發次數：{stop_loss_count}")
    if use_trail or use_atr_trail:
        print(f"   移動停利觸發次數：{trail_stop_count}")
    return position


# ═══════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════

def run_backtest(args):
    # ── --best 一鍵套用最佳策略（覆蓋個別參數）──
    if args.best:
        args.adx = True
        args.adx_period = 8        # ADX(8) 回測最佳天期
        args.relax_entry = True
        args.stop_loss = 0.06
        args.trail_stop = 0.15
        args.min_hold = 5
        args.exit_ma = -2          # 純移動停利出場
        args._rsi_filter = True    # 內部旗標：啟用 RSI 50-80 濾鏡
        args.dist_filter = True    # 高檔出貨過濾
    else:
        args._rsi_filter = False

    print("=" * 56)
    if args.best:
        print("  朱家泓進場法 FinLab 回測（⭐ 最佳策略）")
    else:
        print("  朱家泓進場法 FinLab 回測（v2 優化版）")
    print("=" * 56)
    print(f"  回測年數: {args.years}  |  最大持股: {args.nstocks}")
    stop_label = "ATR×2" if args.atr_stop else f"{args.stop_loss*100:.0f}%"
    print(f"  停損: {stop_label}  |  量確認倍數: {args.vol_mult}x")
    print(f"  乖離率上限: {args.max_bias*100:.0f}%  |  大盤濾網: {'OFF' if args.no_market_filter else 'ON'}")
    trend_label = f"ADX({args.adx_period})趨勢" if args.adx else f"Swing({args.lookback}日)"
    print(f"  趨勢偵測: {trend_label}  |  最低持有: {args.min_hold} 天")
    relax_label = "ON（量1.2x+乖離5%）" if args.relax_entry else "OFF"
    ma_label = "3線(MA5>MA10>MA20)" if args.no_ma60 else "4線(含MA60)"
    trail_label = f"{args.trail_stop*100:.0f}%" if args.trail_stop else "OFF"
    rsi_label = "ON（50-80）" if args._rsi_filter else "OFF"
    print(f"  放寬進場: {relax_label}  |  ATR停損: {'ON' if args.atr_stop else 'OFF'}")
    print(f"  均線多排: {ma_label}  |  移動停利: {trail_label}")
    dist_label = "ON（長黑K+位階>70%+法人賣超）" if args.dist_filter else "OFF"
    print(f"  RSI濾鏡: {rsi_label}  |  高檔出貨: {dist_label}")
    print("=" * 56)
    print()

    # ── 1. 登入 FinLab ──
    import finlab
    from finlab import data

    from config import FINLAB_API_TOKEN
    if not FINLAB_API_TOKEN:
        print("❌ 請先設定 FINLAB_API_TOKEN（config.py 或環境變數）")
        sys.exit(1)

    print("🔑 登入 FinLab...")
    finlab.login(FINLAB_API_TOKEN)

    # 使用 FileStorage 持久化存檔（避免重複消耗 API 用量）
    from finlab.data.storage import FileStorage
    data.set_storage(FileStorage(path="finlab_db"))

    # ── 2. 取得資料（FileStorage 自動快取到 finlab_db/）──
    print("📥 下載歷史資料...")
    close = data.get("price:收盤價")
    volume = data.get("price:成交股數")
    open_ = data.get("price:開盤價")
    high = data.get("price:最高價")
    low = data.get("price:最低價")

    # 只保留 4 碼一般股票（排除 ETF、權證等）
    stock_mask = close.columns.str.match(r"^[1-9]\d{3}$")
    close = close.loc[:, stock_mask]
    volume = volume.loc[:, stock_mask]
    open_ = open_.loc[:, open_.columns.str.match(r"^[1-9]\d{3}$")]
    high = high.loc[:, high.columns.str.match(r"^[1-9]\d{3}$")]
    low = low.loc[:, low.columns.str.match(r"^[1-9]\d{3}$")]

    # 對齊欄位
    common_cols = close.columns.intersection(volume.columns).intersection(
        open_.columns).intersection(high.columns).intersection(low.columns)
    close = close[common_cols]
    volume = volume[common_cols]
    open_ = open_[common_cols]
    high = high[common_cols]
    low = low[common_cols]

    # 自訂股票清單
    if args.stocks:
        stock_list = [s.strip() for s in args.stocks.split(",") if s.strip()]
        valid = [s for s in stock_list if s in close.columns]
        missing = [s for s in stock_list if s not in close.columns]
        if missing:
            print(f"   ⚠️ 以下股票代碼找不到資料：{', '.join(missing)}")
        if not valid:
            print("❌ 自訂清單中沒有任何有效股票")
            sys.exit(1)
        close = close[valid]
        volume = volume[valid]
        open_ = open_[valid]
        high = high[valid]
        low = low[valid]
        print(f"   📋 使用自訂股票清單：{', '.join(valid)}（共 {len(valid)} 檔）")

    if close.empty:
        print("❌ 無法取得任何股價資料，請檢查 FinLab 帳號")
        sys.exit(1)

    # 自動偵測資料的實際日期範圍
    data_start = close.index[0]
    data_end = close.index[-1]
    print(f"   FinLab 資料範圍：{data_start.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")

    # 回測期間：從資料末端往回算 years 年（而非從今天）
    start_date = data_end - pd.DateOffset(years=args.years)

    if start_date < data_start:
        start_date = data_start + pd.DateOffset(days=120)  # 至少保留 120 天給 MA60
        actual_years = (data_end - start_date).days / 365.25
        print(f"   ⚠️ 資料不足 {args.years} 年，自動調整為 {actual_years:.1f} 年")

    # 多抓 120 天讓 MA60 有足夠歷史
    fetch_start = start_date - pd.DateOffset(days=120)
    close = close[close.index >= fetch_start]
    volume = volume[volume.index >= fetch_start]
    open_ = open_[open_.index >= fetch_start]
    high = high[high.index >= fetch_start]
    low = low[low.index >= fetch_start]

    # 檢查是否使用免費版（資料非最新）
    days_behind = (pd.Timestamp.now() - data_end).days
    if days_behind > 30:
        print(f"   ⚠️ 免費版資料僅到 {data_end.strftime('%Y-%m-%d')}（落後 {days_behind} 天）")
        print(f"      升級 FinLab VIP 可取得即時資料")

    print(f"   回測期間：{start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
    print(f"   交易日數：{len(close)}  |  股票檔數：{len(close.columns)}")
    print()

    # ── 3. 取得大盤資料（加權指數）──
    market_filter = None
    if not args.no_market_filter:
        print("📈 取得大盤資料...")
        try:
            benchmark = data.get("benchmark_return:發行量加權股價報酬指數")
            if isinstance(benchmark, pd.DataFrame):
                bench_series = benchmark.iloc[:, 0]
            else:
                bench_series = benchmark

            bench_ma20 = bench_series.rolling(20).mean()
            market_bull = bench_series > bench_ma20  # Series: date → bool

            # 展開成與 close 同形的 DataFrame
            market_filter = pd.DataFrame(
                {col: market_bull for col in close.columns},
                index=close.index,
            ).reindex(close.index).fillna(False)

            bull_days = market_bull.reindex(close.index).fillna(False)
            bull_pct = bull_days.sum() / len(bull_days) * 100
            print(f"   大盤多頭天數比例：{bull_pct:.1f}%")
        except Exception as e:
            print(f"   ⚠️ 大盤資料取得失敗：{e}，跳過大盤濾網")
            market_filter = None

    # ── 3b. 基本面濾網（EPS > 0 且 營收 YoY > 0）──
    fundamental_filter = None
    if not args.no_fundamental:
        print("📊 取得基本面資料...")
        try:
            # EPS（季頻 → 用 FinLab deadline() 轉日頻，避免前視偏差）
            eps_q = data.get("financial_statement:每股盈餘")
            eps_q = eps_q[eps_q.columns.intersection(close.columns)]
            eps_daily = eps_q.deadline()
            eps_daily = eps_daily.reindex(close.index, method="ffill")
            cond_eps = eps_daily > 0

            # 月營收 → 計算 YoY → forward-fill 到日頻
            rev_m = data.get("monthly_revenue:當月營收")
            rev_m = rev_m[rev_m.columns.intersection(close.columns)]
            rev_yoy = rev_m / rev_m.shift(12) - 1
            rev_yoy_daily = rev_yoy.reindex(close.index, method="ffill")
            rev_yoy_threshold = getattr(args, "rev_yoy", 0)
            cond_rev = rev_yoy_daily > rev_yoy_threshold

            fundamental_filter = (cond_eps & cond_rev).reindex(
                index=close.index, columns=close.columns
            ).fillna(False)

            if rev_yoy_threshold > 0:
                print(f"  📊 營收 YoY 門檻：> {rev_yoy_threshold*100:.0f}%")
            pass_pct = fundamental_filter.sum().sum() / fundamental_filter.size * 100
            print(f"   基本面通過率：{pass_pct:.1f}%（EPS>0 且 營收YoY>0）")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   ⚠️ 基本面資料取得失敗：{e}，跳過基本面濾網")
            fundamental_filter = None

    # ── 4. 計算技術指標 ──
    print("📐 計算均線與量能指標...")
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vma5 = volume.rolling(5).mean()

    # ATR(14) 動態停損用
    atr_df = None
    if args.atr_stop:
        print("📐 計算 ATR(14)...")
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=0).groupby(level=0).max()
        # true_range 可能因 concat 方式不對，改用 DataFrame.max
        true_range = pd.DataFrame({
            "tr1": (high - low).stack(),
            "tr2": (high - close.shift(1)).abs().stack(),
            "tr3": (low - close.shift(1)).abs().stack(),
        }).max(axis=1).unstack()
        true_range = true_range.reindex(columns=close.columns)
        atr_df = true_range.rolling(14).mean()
        print(f"   ATR(14) 計算完成")

    # ── 5. 條件 1：趨勢/擺盪偵測 ──
    t0 = time.time()
    if args.adx:
        # ADX 趨勢強度（最佳策略，替代擺盪偵測）
        _adx_p = args.adx_period
        print(f"📐 計算 ADX({_adx_p}) 趨勢強度...")
        _tr1 = high - low
        _tr2 = (high - close.shift(1)).abs()
        _tr3 = (low - close.shift(1)).abs()
        _true_range = pd.DataFrame({
            "tr1": _tr1.stack(), "tr2": _tr2.stack(), "tr3": _tr3.stack()
        }).max(axis=1).unstack().reindex(columns=close.columns)
        _atr = _true_range.rolling(_adx_p).mean()

        _up_move = high - high.shift(1)
        _down_move = low.shift(1) - low
        _plus_dm = pd.DataFrame(np.where((_up_move > _down_move) & (_up_move > 0), _up_move, 0),
                                index=close.index, columns=close.columns)
        _minus_dm = pd.DataFrame(np.where((_down_move > _up_move) & (_down_move > 0), _down_move, 0),
                                 index=close.index, columns=close.columns)
        _plus_di = 100 * (_plus_dm.rolling(_adx_p).mean() / _atr)
        _minus_di = 100 * (_minus_dm.rolling(_adx_p).mean() / _atr)
        _dx = (_plus_di - _minus_di).abs() / (_plus_di + _minus_di).replace(0, np.nan) * 100
        _adx = _dx.rolling(_adx_p).mean()

        _adx_filter = (_adx > 20).fillna(False)
        _uptrend = (_plus_di > _minus_di).fillna(False)
        cond1 = _adx_filter & _uptrend

        elapsed = time.time() - t0
        pass_pct = cond1.sum().sum() / cond1.size * 100
        print(f"   ADX 趨勢偵測完成（{elapsed:.1f}s），通過率：{pass_pct:.1f}%")
    else:
        cond1 = compute_swing_pattern(close, lookback=args.lookback)
        elapsed = time.time() - t0
        print(f"   擺盪偵測完成（{elapsed:.1f}s）")
    print()

    # ── 6. 合成進場訊號（共用）──
    entry_mode = "放寬模式（量1.2x+乖離5%）" if args.relax_entry else "攻擊量 + 基本面模式"
    print(f"🎯 合成進場訊號（{entry_mode}）...")
    entries = build_entry_signals(close, volume, ma5, ma10, ma20, ma60, vma5,
                                  cond1, vol_mult=args.vol_mult,
                                  market_filter=market_filter,
                                  max_bias=args.max_bias,
                                  open_=open_, high=high, low=low,
                                  fundamental_filter=fundamental_filter,
                                  relax_entry=args.relax_entry,
                                  no_ma60=args.no_ma60)

    # ── 6b. RSI(14) 濾鏡（--best 模式啟用）──
    if args._rsi_filter:
        print("📐 計算 RSI(14) 濾鏡...")
        _delta = close.diff()
        _gain = _delta.clip(lower=0).rolling(14).mean()
        _loss = (-_delta.clip(upper=0)).rolling(14).mean()
        _rs = _gain / _loss.replace(0, np.nan)
        _rsi = 100 - (100 / (1 + _rs))
        rsi_mask = (_rsi > 50) & (_rsi < 80)
        rsi_mask = rsi_mask.reindex(index=entries.index, columns=entries.columns).fillna(False)
        before_rsi = int(entries.sum().sum())
        entries = entries & rsi_mask
        after_rsi = int(entries.sum().sum())
        print(f"   RSI 50-80 濾鏡：{before_rsi} → {after_rsi} 訊號（過濾 {before_rsi - after_rsi}）")

    # ── 6c. 高檔出貨過濾（長黑K + 位階>70% + 法人賣超）──
    if args.dist_filter:
        print("📐 計算高檔出貨過濾...")
        # 條件 1：長黑K（收黑 + 實體 > 3%）
        _is_black = close < open_
        _body_pct = (open_ - close) / open_ * 100
        _long_black = _is_black & (_body_pct > 3)

        # 條件 2：位階 > 70%（近 60 日高低區間）
        _high_60 = high.rolling(60).max()
        _low_60 = low.rolling(60).min()
        _range_60 = _high_60 - _low_60
        _position = ((close - _low_60) / _range_60.replace(0, np.nan) * 100).fillna(50)
        _high_pos = _position > 70

        # 條件 3：法人賣超（外資或投信任一淨賣超）
        try:
            print("   📥 下載法人買賣超資料...")
            _foreign_net = data.get("institutional_investors_trading_summary:外陸資買賣超股數(不含外資自營商)")
            _trust_net = data.get("institutional_investors_trading_summary:投信買賣超股數")
            # 對齊到 close 的 index/columns
            _foreign_net = _foreign_net.reindex(index=close.index, columns=close.columns).fillna(0)
            _trust_net = _trust_net.reindex(index=close.index, columns=close.columns).fillna(0)
            _inst_selling = (_foreign_net < 0) | (_trust_net < 0)

            # 模式 A：高檔出貨（長黑K + 位階>70% + 法人賣超）
            _dist_mode_a = _long_black & _high_pos & _inst_selling

            # 模式 B：誘多翻臉（長黑K + 前日買超→今日賣超≥前日買超，不限位階）
            _prev_foreign = _foreign_net.shift(1)
            _prev_trust = _trust_net.shift(1)
            _foreign_flip = (_prev_foreign > 0) & (_foreign_net < 0) & (_foreign_net.abs() >= _prev_foreign)
            _trust_flip = (_prev_trust > 0) & (_trust_net < 0) & (_trust_net.abs() >= _prev_trust)
            _dist_mode_b = _long_black & (_foreign_flip | _trust_flip)

            # 任一模式成立 → 排除
            _dist_top = (_dist_mode_a | _dist_mode_b).fillna(False)
            _dist_top = _dist_top.reindex(index=entries.index, columns=entries.columns).fillna(False)

            before_dist = int(entries.sum().sum())
            entries = entries & (~_dist_top)
            after_dist = int(entries.sum().sum())
            mode_a_count = int(_dist_mode_a.reindex(index=entries.index, columns=entries.columns).fillna(False).sum().sum())
            mode_b_count = int(_dist_mode_b.reindex(index=entries.index, columns=entries.columns).fillna(False).sum().sum())
            print(f"   高檔出貨過濾：{before_dist} → {after_dist} 訊號（排除 {before_dist - after_dist}）")
            print(f"     模式A（高檔+法人賣）：{mode_a_count}  模式B（誘多翻臉）：{mode_b_count}")
        except Exception as e:
            print(f"   ⚠️ 法人資料下載失敗：{e}，跳過高檔出貨過濾")

    # 只取回測期間
    entries = entries[entries.index >= start_date]

    entry_count = int(entries.sum().sum())
    print(f"   進場訊號總數：{entry_count}")

    if entry_count == 0:
        print("❌ 沒有產生任何進場訊號，請檢查策略參數")
        sys.exit(1)

    # 用量比當排序依據
    vol_ratio = volume / vma5
    vol_ratio = vol_ratio[vol_ratio.index >= start_date]

    # ── 判斷模式 ──
    if args.compare:
        # A/B/C/D 四組對比模式
        return _run_comparison(
            entries, close, ma5, ma10, ma20,
            vol_ratio, start_date, data_end, args,
            atr_df=atr_df
        )
    else:
        # 單一模式
        return _run_single(
            entries, close, ma5, ma10, ma20,
            vol_ratio, start_date, data_end, args,
            atr_df=atr_df
        )


def _run_single(entries, close, ma5, ma10, ma20,
                vol_ratio, start_date, data_end, args,
                atr_df=None):
    """執行單一出場策略回測"""

    exit_labels = {5: "MA5 短線", 10: "MA10 慣性", 20: "MA20 波段", 0: "動態模式"}
    label = exit_labels.get(args.exit_ma, f"MA{args.exit_ma}")

    print(f"\n🎯 出場策略：{label}")
    exits = build_exit_signals(close, ma5, ma10, ma20, exit_ma=args.exit_ma)
    exits = exits[exits.index >= start_date]

    print("📦 建構持倉矩陣...")
    position = _build_position(entries, exits, vol_ratio, close,
                               nstocks=args.nstocks,
                               stop_loss=args.stop_loss,
                               min_hold_days=args.min_hold,
                               atr_df=atr_df if args.atr_stop else None,
                               trail_stop=args.trail_stop,
                               full_invest=args.full_invest)

    if position.sum().sum() == 0:
        print("❌ 持倉矩陣全空")
        return None

    # 嘗試使用 FinLab sim()，失敗則自行計算
    report = None
    try:
        print("🚀 執行回測模擬...")
        from finlab.backtest import sim
        report = sim(
            position, resample="D", trade_at_price="close",
            fee_ratio=1.425 / 1000, tax_ratio=3 / 1000,
            name=f"朱家泓-{label}", upload=False,
        )
    except Exception as e:
        print(f"  ⚠️ FinLab sim() 失敗：{e}")
        print("  📊 改用自行計算模式...")

    print()
    print("=" * 56)
    print(f"  📊 朱家泓進場法 — {label}")
    print("=" * 56)
    print(f"  回測期間:     {start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")

    if report is not None:
        _print_report_stats(report)
    else:
        # 自行計算
        stats = _calc_stats_from_position(position, close, start_date, label)
        print(f"  年化報酬率:   {stats['annual_ret']:+.2f}%")
        print(f"  夏普比率:     {stats['sharpe']:.2f}")
        print(f"  最大回撤:     {stats['max_dd']:.2f}%")

    _print_trade_stats(position, close, start_date)
    print()
    return report


def _run_comparison(entries, close, ma5, ma10, ma20,
                    vol_ratio, start_date, data_end, args,
                    atr_df=None):
    """A/B/C/D × 停損對比（自行計算，不需 sim()）"""

    exit_groups = [
        ("A-MA5", 5),
        ("B-MA10", 10),
        ("C-MA20", 20),
        ("D-動態", 0),
    ]
    # 停損模式：ATR 動態 or 固定比例 5%/8%
    if args.atr_stop:
        stop_losses = [("ATR×2", 0.08)]  # ATR 模式只跑一組，fallback 用 8%
    else:
        stop_losses = [("SL5%", 0.05), ("SL8%", 0.08)]

    results = []

    for sl_label, sl in stop_losses:
        for exit_label, exit_ma in exit_groups:
            label = f"{exit_label} {sl_label}"

            print(f"\n{'─'*56}")
            print(f"  🔬 {label}")
            print(f"{'─'*56}")

            exits = build_exit_signals(close, ma5, ma10, ma20, exit_ma=exit_ma)
            exits = exits[exits.index >= start_date]

            print("  📦 建構持倉...")
            position = _build_position(entries, exits, vol_ratio, close,
                                       nstocks=args.nstocks,
                                       stop_loss=sl,
                                       min_hold_days=args.min_hold,
                                       atr_df=atr_df if args.atr_stop else None,
                                       trail_stop=args.trail_stop,
                                       full_invest=args.full_invest)

            if position.sum().sum() == 0:
                print("  ❌ 無持倉")
                results.append({"label": label, "error": True})
                continue

            stats = _calc_stats_from_position(position, close, start_date, label)
            # 保存原始交易明細供 watch 分析
            stats["_trades"] = _extract_trades(position, close)
            results.append(stats)
            print(f"  ✅ {label}: 勝率 {stats['win_rate']:.1f}% | "
                  f"年化 {stats['annual_ret']:+.1f}% | 交易 {stats['total_trades']}")

    # ── 輸出對比報表 ──
    _print_comparison_table(results, start_date, data_end)

    # ── 關注清單交易分析 ──
    if args.watch:
        watch_list = [s.strip() for s in args.watch.split(",") if s.strip()]
        _print_watch_report(results, watch_list, start_date, data_end)

    return results


def _calc_stats_from_position(position, close, start_date, label):
    """
    從持倉矩陣 + 收盤價自行計算完整績效指標。
    不需要 FinLab sim()，節省 API 用量。
    """
    stats = {"label": label, "error": False}

    # ── 計算每日組合報酬 ──
    # 報酬 = Σ(weight_i × daily_return_i)
    daily_ret = close.pct_change().fillna(0)
    # position 在 t 日的權重 × t+1 日的報酬（隔日才實現）
    portfolio_ret = (position.shift(1) * daily_ret).sum(axis=1)
    portfolio_ret = portfolio_ret[portfolio_ret.index >= start_date]

    # 扣除交易成本（區分買入/賣出）
    # 台股：買入手續費 0.1425%（券商6折=0.0855%）、賣出手續費 + 證交稅 0.3%
    buy_fee = 1.425 / 1000 * 0.6    # 0.0855%（手續費6折）
    sell_fee = 1.425 / 1000 * 0.6 + 3 / 1000  # 0.3855%（手續費6折+證交稅）
    pos_diff = position.diff()
    buy_turnover = pos_diff.clip(lower=0).sum(axis=1)     # 增加持倉 = 買入
    sell_turnover = (-pos_diff.clip(upper=0)).sum(axis=1)  # 減少持倉 = 賣出
    daily_cost = buy_turnover * buy_fee + sell_turnover * sell_fee
    portfolio_ret = portfolio_ret - daily_cost

    # 累積報酬
    equity = (1 + portfolio_ret).cumprod()

    if len(equity) > 1:
        total_ret = equity.iloc[-1] - 1
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        stats["annual_ret"] = ((1 + total_ret) ** (1 / years) - 1) * 100 if years > 0 else 0

        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        stats["max_dd"] = abs(drawdown.min()) * 100

        pr = portfolio_ret[portfolio_ret != 0]
        stats["sharpe"] = float(pr.mean() / pr.std() * np.sqrt(252)) if pr.std() > 0 else 0
    else:
        stats["annual_ret"] = stats["max_dd"] = stats["sharpe"] = 0

    # ── 交易統計 ──
    trades = _extract_trades(position, close)
    total = len(trades)
    stats["total_trades"] = total

    if total > 0:
        trades_df = pd.DataFrame(trades)
        wins = int((trades_df["return"] > 0).sum())
        losses = total - wins
        stats["win_rate"] = wins / total * 100
        stats["wins"] = wins
        stats["losses"] = losses
        stats["avg_ret"] = trades_df["return"].mean() * 100
        stats["avg_win"] = trades_df.loc[trades_df["return"] > 0, "return"].mean() * 100 if wins > 0 else 0
        stats["avg_loss"] = trades_df.loc[trades_df["return"] <= 0, "return"].mean() * 100 if losses > 0 else 0
        stats["avg_days"] = trades_df["days"].mean()
        stats["pl_ratio"] = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else 0
    else:
        stats.update({"win_rate": 0, "wins": 0, "losses": 0, "avg_ret": 0,
                      "avg_win": 0, "avg_loss": 0, "avg_days": 0, "pl_ratio": 0})

    return stats


def _collect_stats(report, position, close, start_date, label):
    """收集單一實驗組的統計數據"""
    stats = {"label": label, "error": False}

    # FinLab 報告統計 — 使用 get_metrics()
    stats["annual_ret"] = stats["sharpe"] = stats["max_dd"] = 0
    try:
        metrics = report.get_metrics() if hasattr(report, "get_metrics") else {}
        if metrics:
            prof = metrics.get("profitability", {})
            risk = metrics.get("risk", {})
            ratio = metrics.get("ratio", {})

            stats["annual_ret"] = float(prof.get("annualReturn", 0)) * 100
            stats["max_dd"] = abs(float(risk.get("maxDrawdown", 0))) * 100
            stats["sharpe"] = float(ratio.get("sharpeRatio", 0))

        # 備用：從 creturn 累積報酬計算
        if stats["annual_ret"] == 0 and hasattr(report, "creturn"):
            equity = report.creturn
            if isinstance(equity, pd.Series) and len(equity) > 1:
                total_ret = equity.iloc[-1] / equity.iloc[0] - 1
                years = (equity.index[-1] - equity.index[0]).days / 365.25
                if years > 0:
                    stats["annual_ret"] = ((1 + total_ret) ** (1 / years) - 1) * 100
                rolling_max = equity.cummax()
                drawdown = (equity - rolling_max) / rolling_max
                stats["max_dd"] = abs(drawdown.min()) * 100
                daily_ret = equity.pct_change().dropna()
                stats["sharpe"] = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    except Exception as e:
        print(f"    ⚠️ 統計擷取異常：{e}")

    # 交易統計
    trades = _extract_trades(position, close)
    total = len(trades)
    stats["total_trades"] = total

    if total > 0:
        trades_df = pd.DataFrame(trades)
        wins = int((trades_df["return"] > 0).sum())
        losses = total - wins
        stats["win_rate"] = wins / total * 100
        stats["wins"] = wins
        stats["losses"] = losses
        stats["avg_ret"] = trades_df["return"].mean() * 100
        stats["avg_win"] = trades_df.loc[trades_df["return"] > 0, "return"].mean() * 100 if wins > 0 else 0
        stats["avg_loss"] = trades_df.loc[trades_df["return"] <= 0, "return"].mean() * 100 if losses > 0 else 0
        stats["avg_days"] = trades_df["days"].mean()
        stats["pl_ratio"] = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else 0
    else:
        stats.update({"win_rate": 0, "wins": 0, "losses": 0, "avg_ret": 0,
                      "avg_win": 0, "avg_loss": 0, "avg_days": 0, "pl_ratio": 0})

    return stats


def _extract_trades(position, close):
    """從持倉矩陣抽取交易列表"""
    trades = []
    held = position.astype(bool)

    for sid in held.columns:
        col = held[sid]
        in_trade = False
        entry_date = None

        for date in col.index:
            if col[date] and not in_trade:
                in_trade = True
                entry_date = date
            elif not col[date] and in_trade:
                in_trade = False
                if sid in close.columns:
                    ep = close.loc[entry_date, sid] if entry_date in close.index else np.nan
                    xp = close.loc[date, sid] if date in close.index else np.nan
                    if not np.isnan(ep) and not np.isnan(xp) and ep > 0:
                        trades.append({
                            "stock_id": sid,
                            "entry_date": entry_date,
                            "exit_date": date,
                            "return": (xp / ep) - 1,
                            "days": (date - entry_date).days,
                        })
    return trades


def _print_watch_report(results, watch_list, start_date, data_end):
    """印出關注清單的交易分析報告"""
    valid = [r for r in results if not r.get("error") and "_trades" in r]
    if not valid:
        return

    # 收集所有關注清單的交易
    all_watch_trades = []
    for r in valid:
        matched = [t for t in r["_trades"] if t["stock_id"] in watch_list]
        for t in matched:
            t["group"] = r["label"]
        all_watch_trades.extend(matched)

    # 找出哪些關注股票曾被策略選到
    hit_stocks = sorted(set(t["stock_id"] for t in all_watch_trades))
    miss_stocks = sorted(set(watch_list) - set(hit_stocks))

    print()
    print("=" * 72)
    print("  🔍 關注清單交易分析（策略 G 從全市場選股 → 命中你的持股）")
    print(f"  回測期間：{start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
    print(f"  關注清單：{len(watch_list)} 檔  |  被選到：{len(hit_stocks)} 檔")
    print("=" * 72)

    if miss_stocks:
        print(f"  ❌ 從未被選到的股票：{', '.join(miss_stocks)}")

    if not hit_stocks:
        print("  ⚠️ 策略在回測期間從未選到關注清單中的任何股票")
        print("=" * 72)
        return

    print(f"  ✅ 曾被選到的股票：{', '.join(hit_stocks)}")
    print()

    # ── 以最佳出場策略（第一組 valid）統計各股交易 ──
    # 只用 D-動態 SL5%（或第一個 valid 組）的交易
    best_group = valid[0]
    best_trades = [t for t in best_group["_trades"] if t["stock_id"] in watch_list]

    if not best_trades:
        # 如果最佳組沒有交易，找有交易的第一組
        for r in valid:
            best_trades = [t for t in r["_trades"] if t["stock_id"] in watch_list]
            if best_trades:
                best_group = r
                break

    # ── 各出場策略 × 關注清單彙總 ──
    print(f"  {'出場策略':<26} {'命中交易':>8} {'勝率':>6} {'平均報酬':>8} {'盈虧比':>6}")
    print(f"  {'─'*26} {'─'*8} {'─'*6} {'─'*8} {'─'*6}")

    for r in valid:
        matched = [t for t in r["_trades"] if t["stock_id"] in watch_list]
        n = len(matched)
        if n == 0:
            print(f"  {r['label']:<24} {'0':>8} {'---':>6} {'---':>8} {'---':>6}")
            continue
        tdf = pd.DataFrame(matched)
        wins = int((tdf["return"] > 0).sum())
        losses = n - wins
        wr = wins / n * 100
        avg_r = tdf["return"].mean() * 100
        avg_w = tdf.loc[tdf["return"] > 0, "return"].mean() * 100 if wins > 0 else 0
        avg_l = tdf.loc[tdf["return"] <= 0, "return"].mean() * 100 if losses > 0 else 0
        plr = abs(avg_w / avg_l) if avg_l != 0 else 0
        print(f"  {r['label']:<24} {n:>8} {wr:>5.1f}% {avg_r:>+7.2f}% {plr:>5.2f}")

    print(f"  {'─'*26} {'─'*8} {'─'*6} {'─'*8} {'─'*6}")

    # ── 個股明細（用所有組合的交易）──
    print()
    print("  📝 個股交易明細（所有出場策略合併）：")
    for sid in hit_stocks:
        sid_trades = [t for t in all_watch_trades if t["stock_id"] == sid]
        n = len(sid_trades)
        tdf = pd.DataFrame(sid_trades)
        wins = int((tdf["return"] > 0).sum())
        wr = wins / n * 100
        avg_r = tdf["return"].mean() * 100
        print(f"\n  ── {sid}（{n} 筆交易 | 勝率 {wr:.0f}% | 平均報酬 {avg_r:+.2f}%）──")

        # 依時間排序顯示交易
        tdf = tdf.sort_values("entry_date")
        for _, t in tdf.iterrows():
            ret = t["return"] * 100
            icon = "🟢" if ret > 0 else "🔴"
            ed = t["entry_date"].strftime("%Y-%m-%d")
            xd = t["exit_date"].strftime("%Y-%m-%d")
            grp = t.get("group", "")
            print(f"    {icon} {ed} → {xd}  {ret:+.1f}%  ({t['days']}天) [{grp}]")

    print()
    print("=" * 72)


def _print_comparison_table(results, start_date, data_end):
    """印出四組對比報表"""
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("\n❌ 所有實驗組都失敗")
        return

    print()
    print("=" * 72)
    print("  📊 朱家泓進場法 — A/B/C/D 出場策略對比報表")
    print(f"  回測期間：{start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
    print("=" * 72)

    # 表頭
    print(f"  {'實驗組':<26} {'勝率':>6} {'年化報酬':>8} {'最大回撤':>8} "
          f"{'交易次數':>8} {'盈虧比':>6} {'平均持有':>6}")
    print(f"  {'─'*26} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    for r in valid:
        print(f"  {r['label']:<24} "
              f"{r['win_rate']:>5.1f}% "
              f"{r['annual_ret']:>+7.1f}% "
              f"{r['max_dd']:>7.1f}% "
              f"{r['total_trades']:>8} "
              f"{r['pl_ratio']:>5.2f} "
              f"{r['avg_days']:>5.1f}天")

    print(f"  {'─'*26} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

    # 細部比較
    print()
    print("  📋 細部指標：")
    for r in valid:
        print(f"  {r['label'][:1]}組: "
              f"平均獲利 {r['avg_win']:+.2f}%  "
              f"平均虧損 {r['avg_loss']:+.2f}%  "
              f"平均報酬 {r['avg_ret']:+.2f}%  "
              f"({r['wins']}W/{r['losses']}L)")

    # 結論建議
    print()
    print("  " + "═" * 52)
    print("  💡 結論建議：")

    # 找最佳年化報酬
    best_ret = max(valid, key=lambda x: x["annual_ret"])
    # 找最低回撤
    best_dd = min(valid, key=lambda x: x["max_dd"])
    # 找最高勝率
    best_wr = max(valid, key=lambda x: x["win_rate"])
    # 綜合：年化報酬 / 最大回撤 比值最高
    for r in valid:
        r["ret_dd_ratio"] = r["annual_ret"] / r["max_dd"] if r["max_dd"] > 0 else 0
    best_balance = max(valid, key=lambda x: x["ret_dd_ratio"])

    print(f"  • 最高勝率：{best_wr['label']} ({best_wr['win_rate']:.1f}%)")
    print(f"  • 最高年化報酬：{best_ret['label']} ({best_ret['annual_ret']:+.1f}%)")
    print(f"  • 最低回撤：{best_dd['label']} ({best_dd['max_dd']:.1f}%)")
    print(f"  • 報酬/回撤最佳平衡：{best_balance['label']}")
    print(f"    → 推薦使用此出場策略")
    print("  " + "═" * 52)


# ═══════════════════════════════════════════════════════════
# 全方位優化模式
# ═══════════════════════════════════════════════════════════

def run_optimize(args):
    """
    自動測試多組參數組合，找出最佳配置。
    測試維度：
      - MA 多排：4 線(含MA60) vs 3 線(去MA60)
      - 擺盪回看：40 天 vs 20 天
      - 進場模式：嚴格 vs 放寬
      - 出場策略：D-動態
      - 停損：8%
      - 緩衝期：5 天
      - 移動停利：OFF vs 10% vs 15%
    """
    print("=" * 72)
    print("  🔬 全方位優化模式 — 自動參數掃描")
    print("=" * 72)
    print()

    # ── 1. 登入 + 載入資料（只做一次）──
    import finlab
    from finlab import data
    from config import FINLAB_API_TOKEN

    if not FINLAB_API_TOKEN:
        print("❌ 請先設定 FINLAB_API_TOKEN")
        sys.exit(1)

    finlab.login(FINLAB_API_TOKEN)
    from finlab.data.storage import FileStorage
    data.set_storage(FileStorage(path="finlab_db"))

    print("📥 下載歷史資料...")
    close = data.get("price:收盤價")
    volume = data.get("price:成交股數")
    open_ = data.get("price:開盤價")
    high = data.get("price:最高價")
    low = data.get("price:最低價")

    # 只保留 4 碼一般股票
    stock_mask = close.columns.str.match(r"^[1-9]\d{3}$")
    close = close.loc[:, stock_mask]
    volume = volume.loc[:, stock_mask]
    open_ = open_.loc[:, open_.columns.str.match(r"^[1-9]\d{3}$")]
    high = high.loc[:, high.columns.str.match(r"^[1-9]\d{3}$")]
    low = low.loc[:, low.columns.str.match(r"^[1-9]\d{3}$")]

    common_cols = close.columns.intersection(volume.columns).intersection(
        open_.columns).intersection(high.columns).intersection(low.columns)
    close = close[common_cols]
    volume = volume[common_cols]
    open_ = open_[common_cols]
    high = high[common_cols]
    low = low[common_cols]

    data_end = close.index[-1]
    start_date = data_end - pd.DateOffset(years=args.years)
    data_start = close.index[0]
    if start_date < data_start:
        start_date = data_start + pd.DateOffset(days=120)

    fetch_start = start_date - pd.DateOffset(days=120)
    close = close[close.index >= fetch_start]
    volume = volume[volume.index >= fetch_start]
    open_ = open_[open_.index >= fetch_start]
    high = high[high.index >= fetch_start]
    low = low[low.index >= fetch_start]

    print(f"   回測期間：{start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
    print(f"   股票檔數：{len(close.columns)}")

    # ── 大盤濾網 ──
    market_filter = None
    if not args.no_market_filter:
        try:
            benchmark = data.get("benchmark_return:發行量加權股價報酬指數")
            bench_series = benchmark.iloc[:, 0] if isinstance(benchmark, pd.DataFrame) else benchmark
            bench_ma20 = bench_series.rolling(20).mean()
            market_bull = bench_series > bench_ma20
            market_filter = pd.DataFrame(
                {col: market_bull for col in close.columns}, index=close.index
            ).reindex(close.index).fillna(False)
        except Exception:
            market_filter = None

    # ── 基本面濾網（多門檻版本）──
    fundamental_filter = None
    fund_filters = {}  # {threshold: DataFrame} — 供 YoY 門檻掃描用
    if not args.no_fundamental:
        try:
            eps_q = data.get("financial_statement:每股盈餘")
            eps_q = eps_q[eps_q.columns.intersection(close.columns)]
            eps_daily = eps_q.deadline().reindex(close.index, method="ffill")
            cond_eps = eps_daily > 0
            rev_m = data.get("monthly_revenue:當月營收")
            rev_m = rev_m[rev_m.columns.intersection(close.columns)]
            rev_yoy = rev_m / rev_m.shift(12) - 1
            rev_yoy_daily = rev_yoy.reindex(close.index, method="ffill")

            # 預計算多組 YoY 門檻的 fundamental_filter
            for thr in [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
                cond_rev = rev_yoy_daily > thr
                fund_filters[thr] = (cond_eps & cond_rev).reindex(
                    index=close.index, columns=close.columns).fillna(False)
            # 預設 0% 門檻（向下相容）
            fundamental_filter = fund_filters[0]
            print(f"  📊 基本面濾網：已預計算 {len(fund_filters)} 組 YoY 門檻")
        except Exception:
            fundamental_filter = None

    # ── 計算技術指標 ──
    print("📐 計算技術指標...")
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    vma5 = volume.rolling(5).mean()
    vol_ratio = volume / vma5

    # ── 預計算兩組擺盪型態 ──
    print("📊 計算擺盪型態（40日 + 20日）...")
    t0 = time.time()
    swing_40 = compute_swing_pattern(close, lookback=40)
    swing_20 = compute_swing_pattern(close, lookback=20)
    print(f"   完成（{time.time()-t0:.1f}s）")

    # ── 計算額外進場濾鏡 ──
    print("📐 計算 RSI + 相對強度...")

    # RSI(14) 計算
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # 相對強度：個股 20 日報酬 > 大盤 20 日報酬
    stock_ret20 = close / close.shift(20) - 1
    if market_filter is not None:
        try:
            benchmark = data.get("benchmark_return:發行量加權股價報酬指數")
            bench_s = benchmark.iloc[:, 0] if isinstance(benchmark, pd.DataFrame) else benchmark
            bench_ret20 = bench_s / bench_s.shift(20) - 1
            relative_strength = pd.DataFrame(
                {col: stock_ret20[col] > bench_ret20 for col in close.columns},
                index=close.index,
            ).fillna(False)
        except Exception:
            relative_strength = None
    else:
        relative_strength = None

    # RSI 濾鏡：RSI 50-80（趨勢中但未超買）
    rsi_filter = (rsi > 50) & (rsi < 80)
    rsi_filter = rsi_filter.reindex(index=close.index, columns=close.columns).fillna(False)

    print(f"   RSI 50-80 通過率：{rsi_filter.sum().sum() / rsi_filter.size * 100:.1f}%")
    if relative_strength is not None:
        print(f"   相對強度通過率：{relative_strength.sum().sum() / relative_strength.size * 100:.1f}%")

    # RSI 寬範圍 (45-85)
    rsi_filter_wide = (rsi > 45) & (rsi < 85)
    rsi_filter_wide = rsi_filter_wide.reindex(index=close.index, columns=close.columns).fillna(False)

    # ── 新優化維度 ──
    print("📐 計算動量排名 + 條件記憶...")

    # 1. 動量排名：20 日漲幅（選最強趨勢股）
    momentum_20d = close / close.shift(20) - 1
    # 結合量比+動量的綜合排名（各佔50%）
    vol_rank_pct = vol_ratio.rank(axis=1, pct=True)
    mom_rank_pct = momentum_20d.rank(axis=1, pct=True)
    combo_rank = (vol_rank_pct + mom_rank_pct) / 2

    # 2. 條件記憶：擺盪型態近 N 天內曾為 True 即可（大幅增加訊號）
    swing_40_mem3 = swing_40.rolling(3, min_periods=1).max().fillna(0).astype(bool)
    swing_40_mem5 = swing_40.rolling(5, min_periods=1).max().fillna(0).astype(bool)

    # 3. ADX(14) 趨勢強度濾鏡（替代嚴格擺盪）
    print("📐 計算 ADX(14) 趨勢強度...")
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.DataFrame({
        "tr1": tr1.stack(), "tr2": tr2.stack(), "tr3": tr3.stack()
    }).max(axis=1).unstack().reindex(columns=close.columns)
    atr14 = true_range.rolling(14).mean()

    # +DM / -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.DataFrame(np.where((up_move > down_move) & (up_move > 0), up_move, 0),
                           index=close.index, columns=close.columns)
    minus_dm = pd.DataFrame(np.where((down_move > up_move) & (down_move > 0), down_move, 0),
                            index=close.index, columns=close.columns)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
    adx = dx.rolling(14).mean()

    # ADX > 20 = 趨勢存在
    adx_filter = (adx > 20).fillna(False)
    # +DI > -DI = 上升趨勢
    uptrend_filter = (plus_di > minus_di).fillna(False)
    adx_trend = adx_filter & uptrend_filter

    print(f"   ADX>20 且上升趨勢通過率：{adx_trend.sum().sum() / adx_trend.size * 100:.1f}%")
    print(f"   擺盪記憶3日通過率：{swing_40_mem3.sum().sum() / swing_40_mem3.size * 100:.1f}%")
    print(f"   擺盪記憶5日通過率：{swing_40_mem5.sum().sum() / swing_40_mem5.size * 100:.1f}%")

    # 4. ATR-based trailing stop
    # 用 3×ATR 作為從最高點的回落距離（自適應波動度）
    atr_trail_pct = (3 * atr14 / close).clip(upper=0.25)  # 上限25%

    # ── 第五輪優化（深度調參 C1/C2 冠軍）──
    configs = [
        # === 基線參考 ===
        {"name": "REF:C1冠軍ADX+量排",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "REF:C2亞軍ADX+綜合",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "combo"},

        # === A. C1停損調整 ===
        {"name": "A1:C1+SL10%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.10, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "A2:C1+SL12%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.12, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "A3:C1+SL6%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.06, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        # === B. C1停利調整 ===
        {"name": "B1:C1+停利12%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.12, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "B2:C1+停利18%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.18, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "B3:C1+停利20%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.20, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        # === C. C1緩衝期調整 ===
        {"name": "C3:C1+緩3日",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 3, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "C4:C1+緩10日",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 10, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        # === D. C1濾鏡調整 ===
        {"name": "D1:C1去基本面",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol",
         "no_fund": True},

        {"name": "D2:C1去大盤",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol",
         "no_mkt": True},

        {"name": "D3:C1去RSI",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": False, "rs": False, "rank_mode": "vol"},

        {"name": "D4:C1 RSI寬(45-85)",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": "wide", "rs": False, "rank_mode": "vol"},

        # === E. 最佳停損+停利組合 ===
        {"name": "E1:C1+SL10%停利18%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.18, "sl": 0.10, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        {"name": "E2:C1+SL6%停利12%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.12, "sl": 0.06, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol"},

        # === F. C2深度調參 ===
        {"name": "F1:C2+SL10%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.10, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "combo"},

        {"name": "F2:C2+停利18%",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.18, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "combo"},

        # === G. 持股數調整 ===
        {"name": "G1:C1+8檔",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol",
         "nstocks": 8},

        {"name": "G2:C1+15檔",
         "relax": True, "no_ma60": False, "swing": adx_trend,
         "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
         "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol",
         "nstocks": 15},
    ]

    # === H. 營收 YoY 門檻掃描（基於 C1 冠軍配置）===
    if fund_filters:
        for thr_pct in [0, 5, 10, 15, 20, 25, 30]:
            configs.append({
                "name": f"H:C1+YoY>{thr_pct}%",
                "relax": True, "no_ma60": False, "swing": adx_trend,
                "min_hold": 5, "trail": 0.15, "sl": 0.08, "full_invest": False,
                "exit_ma": -2, "rsi": True, "rs": False, "rank_mode": "vol",
                "fund_filter_key": thr_pct / 100,
            })

    # ── 測試所有配置 ──
    results = []
    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'─'*56}")
        print(f"  🔬 {name}")
        print(f"{'─'*56}")

        # 選擇大盤/基本面濾鏡
        cfg_mkt = None if cfg.get("no_mkt") else market_filter
        if cfg.get("no_fund"):
            cfg_fund = None
        elif cfg.get("fund_filter_key") is not None and fund_filters:
            cfg_fund = fund_filters.get(cfg["fund_filter_key"], fundamental_filter)
        else:
            cfg_fund = fundamental_filter

        # 建構進場訊號
        entries = build_entry_signals(
            close, volume, ma5, ma10, ma20, ma60, vma5,
            cfg["swing"], vol_mult=args.vol_mult,
            market_filter=cfg_mkt, max_bias=args.max_bias,
            open_=open_, high=high, low=low,
            fundamental_filter=cfg_fund,
            relax_entry=cfg["relax"],
            no_ma60=cfg["no_ma60"],
        )

        # 額外進場濾鏡
        rsi_mode = cfg.get("rsi")
        if rsi_mode == "wide":
            entries = entries & rsi_filter_wide
        elif rsi_mode:
            entries = entries & rsi_filter
        if cfg.get("rs") and relative_strength is not None:
            entries = entries & relative_strength

        entries = entries[entries.index >= start_date]
        entry_count = int(entries.sum().sum())
        print(f"  進場訊號：{entry_count}")

        if entry_count == 0:
            results.append({"label": name, "error": True})
            continue

        # 出場策略
        exit_ma_mode = cfg.get("exit_ma", 0)
        exits = build_exit_signals(close, ma5, ma10, ma20, exit_ma=exit_ma_mode)
        exits = exits[exits.index >= start_date]

        # 選擇排名方式
        rank_mode = cfg.get("rank_mode", "vol")
        if rank_mode == "momentum":
            rank_df = momentum_20d
        elif rank_mode == "combo":
            rank_df = combo_rank
        else:
            rank_df = vol_ratio
        vr = rank_df[rank_df.index >= start_date]

        # 處理 ATR 自適應停利
        trail_val = cfg["trail"]
        trail_atr_df = None
        if trail_val == "atr":
            trail_val = None  # _build_position 裡用 atr_trail_pct 替代
            trail_atr_df = atr_trail_pct

        # 建構持倉
        cfg_nstocks = cfg.get("nstocks", args.nstocks)
        position = _build_position(
            entries, exits, vr, close,
            nstocks=cfg_nstocks, stop_loss=cfg["sl"],
            min_hold_days=cfg["min_hold"],
            trail_stop=trail_val,
            full_invest=cfg.get("full_invest", False),
            atr_trail_df=trail_atr_df,
        )

        if position.sum().sum() == 0:
            results.append({"label": name, "error": True})
            continue

        stats = _calc_stats_from_position(position, close, start_date, name)
        stats["entry_count"] = entry_count
        results.append(stats)
        print(f"  ✅ 年化 {stats['annual_ret']:+.1f}% | 勝率 {stats['win_rate']:.1f}% | "
              f"交易 {stats['total_trades']} | MDD {stats['max_dd']:.1f}%")

    # ── 輸出總表 ──
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print("\n❌ 所有配置都失敗")
        return

    print()
    print("=" * 90)
    print("  🏆 全方位優化結果總表")
    print(f"  回測期間：{start_date.strftime('%Y-%m-%d')} ~ {data_end.strftime('%Y-%m-%d')}")
    print("=" * 90)

    print(f"  {'配置':<26} {'訊號數':>6} {'交易':>6} {'勝率':>6} {'年化':>8} "
          f"{'MDD':>8} {'盈虧比':>6} {'均報酬':>8} {'均天數':>6}")
    print(f"  {'─'*26} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*6}")

    for r in sorted(valid, key=lambda x: x["annual_ret"], reverse=True):
        marker = " ⭐" if r["annual_ret"] == max(v["annual_ret"] for v in valid) else ""
        entry_c = r.get("entry_count", "?")
        print(f"  {r['label']:<24} "
              f"{entry_c:>6} "
              f"{r['total_trades']:>6} "
              f"{r['win_rate']:>5.1f}% "
              f"{r['annual_ret']:>+7.1f}% "
              f"{r['max_dd']:>7.1f}% "
              f"{r['pl_ratio']:>5.2f} "
              f"{r['avg_ret']:>+7.2f}% "
              f"{r['avg_days']:>5.1f}天{marker}")

    print(f"  {'─'*26} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*6}")

    # 最佳配置
    best = max(valid, key=lambda x: x["annual_ret"])
    print()
    print(f"  🏆 最佳年化報酬：{best['label']}")
    print(f"     年化: {best['annual_ret']:+.1f}% | 勝率: {best['win_rate']:.1f}% | "
          f"MDD: {best['max_dd']:.1f}% | 盈虧比: {best['pl_ratio']:.2f}")
    print(f"     交易 {best['total_trades']} 筆 | 平均報酬 {best['avg_ret']:+.2f}% | "
          f"平均持有 {best['avg_days']:.1f} 天")

    # 報酬為正的配置
    positive = [r for r in valid if r["annual_ret"] > 0]
    if positive:
        print(f"\n  ✅ 年化報酬為正的配置有 {len(positive)} 個：")
        for r in sorted(positive, key=lambda x: x["annual_ret"], reverse=True):
            print(f"     • {r['label']}: {r['annual_ret']:+.1f}%")
    else:
        print(f"\n  ⚠️ 尚無年化報酬為正的配置")
        # 提供下一步建議
        print(f"     但最佳配置已從 -10.9% 改善到 {best['annual_ret']:+.1f}%")
        if best['avg_ret'] > 0:
            print(f"     每筆交易平均報酬已為正（{best['avg_ret']:+.2f}%），主要瓶頸是閒置資金")

    print("=" * 90)
    return results


# ═══════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.optimize:
        run_optimize(args)
    else:
        run_backtest(args)
