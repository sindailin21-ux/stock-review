"""
strategies/__init__.py
策略註冊中心 — 所有策略模組在 import 時自動註冊。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class StrategyInfo:
    """策略描述資料。"""
    code: str                          # "A"~"G"
    name: str                          # "均線糾結起漲"
    description: str                   # UI tooltip
    category: str                      # "classic" | "master_chu"
    pool: Optional[str]                # "A", "B", or None
    needs_institutional: bool          # 是否需要法人資料
    needs_industry: bool               # 是否需要產業字串
    css_class: str                     # "strat-a" ~ "strat-g"
    func: Callable                     # 進場策略函式 (df, **kwargs) -> dict | None
    has_review_mode: bool = False      # 是否具有覆盤模式
    review_func: Optional[Callable] = None  # 覆盤函式 (df, **kwargs) -> dict


# ── 全域策略註冊表 ──
_REGISTRY: dict[str, StrategyInfo] = {}


def register(info: StrategyInfo) -> StrategyInfo:
    """註冊一個策略（在模組 import 時呼叫）。"""
    _REGISTRY[info.code] = info
    return info


def get_strategy(code: str) -> Optional[StrategyInfo]:
    """依代碼取得策略資訊。"""
    return _REGISTRY.get(code)


def get_all_strategies() -> dict[str, StrategyInfo]:
    """取得所有已註冊策略。"""
    return dict(_REGISTRY)


def get_strategies_by_category(category: str) -> list[StrategyInfo]:
    """依分類取得策略列表。"""
    return [s for s in _REGISTRY.values() if s.category == category]


def discover_strategies():
    """匯入所有策略模組以觸發 register() 呼叫。"""
    from strategies import classic       # noqa: F401
    from strategies import master_chu    # noqa: F401
