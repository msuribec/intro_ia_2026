from __future__ import annotations

from typing import Any, NotRequired, Optional, TypedDict

import pandas as pd


class ReceiptItem(TypedDict):
    name: str
    price: float
    category: str


class ReceiptData(TypedDict, total=False):
    vendor: str
    date: str
    currency: str
    items: list[ReceiptItem]
    total: float
    category_totals: dict[str, float]
    savings_tip: str
    tip_language: str


class ReceiptEntry(TypedDict, total=False):
    image_bytes: Optional[bytes]
    source: NotRequired[str]
    data: ReceiptData
    audio_bytes: Optional[bytes]


class CategorySuggestion(TypedDict, total=False):
    category: str
    reason: str


class HistoryTipEntry(TypedDict, total=False):
    tip: str
    audio_bytes: Optional[bytes]
    tip_language: str


class HistorySummary(TypedDict):
    receipts: list[dict[str, Any]]
    items: list[dict[str, Any]]
    top_stores: dict[str, int]
    category_totals: dict[str, float]


class BudgetSource(TypedDict):
    items_df: pd.DataFrame
    receipts_df: pd.DataFrame
    skipped_receipts: int


class BudgetAnalytics(TypedDict):
    month_period: pd.Period
    month_receipts_df: pd.DataFrame
    month_items_df: pd.DataFrame
    category_df: pd.DataFrame
    stacked_df: pd.DataFrame
    daily_df: pd.DataFrame
    total_budget: float
    total_actual: float
    currency: str


class RagMessage(TypedDict):
    role: str
    content: str

