from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional

import streamlit as st

from expense_auditor.types import (
    CategorySuggestion,
    HistoryTipEntry,
    RagMessage,
    ReceiptData,
    ReceiptEntry,
)


SESSION_DEFAULTS = {
    "widget_seed": 0,
    "categories_approved": False,
    "categories_signature": "",
    "approved_categories": [],
    "receipt_history": [],
    "editing_receipt_index": None,
    "history_category_suggestions": [],
    "history_purchase_tips": [],
    "category_budgets": {},
    "selected_budget_month": "",
    "vector_store": None,
    "rag_chat_history": [],
    "rag_receipt_count": 0,
}


class SessionRepository:
    """Single access point for Streamlit session state."""

    def __init__(self) -> None:
        self._state = st.session_state

    def bootstrap(self) -> None:
        for key, value in SESSION_DEFAULTS.items():
            if key not in self._state:
                self._state[key] = deepcopy(value)

    def restart_streamlit_app(self) -> None:
        st.cache_data.clear()
        st.cache_resource.clear()
        next_seed = self.get_widget_seed() + 1
        for key in list(self._state.keys()):
            del self._state[key]
        self._state["widget_seed"] = next_seed
        st.rerun()

    def get_widget_seed(self) -> int:
        return int(self._state.get("widget_seed", 0))

    def get_categories_approved(self) -> bool:
        return bool(self._state.get("categories_approved", False))

    def set_categories_approved(self, value: bool) -> None:
        self._state["categories_approved"] = bool(value)

    def get_categories_signature(self) -> str:
        return str(self._state.get("categories_signature", ""))

    def set_categories_signature(self, signature: str) -> None:
        self._state["categories_signature"] = signature

    def get_approved_categories(self) -> list[str]:
        return list(self._state.get("approved_categories", []))

    def set_approved_categories(self, categories: list[str]) -> None:
        self._state["approved_categories"] = list(categories)

    def approve_categories(self, categories: list[str], signature: str) -> None:
        self.set_categories_approved(True)
        self.set_categories_signature(signature)
        self.set_approved_categories(categories)

    def invalidate_categories(self, signature: str = "") -> None:
        self.set_categories_signature(signature)
        self.set_categories_approved(False)
        self.set_approved_categories([])

    def get_receipt_history(self) -> list[ReceiptEntry]:
        return list(self._state.get("receipt_history", []))

    def append_receipt(self, entry: ReceiptEntry) -> None:
        self._state["receipt_history"].append(entry)
        self.invalidate_rag_index()

    def extend_receipt_history(self, entries: list[ReceiptEntry]) -> None:
        self._state["receipt_history"].extend(entries)
        self.invalidate_rag_index()

    def update_receipt_data(self, index: int, data: ReceiptData) -> None:
        self._state["receipt_history"][index]["data"] = data
        self.invalidate_rag_index()

    def update_receipt_audio(self, index: int, audio_bytes: Optional[bytes]) -> None:
        self._state["receipt_history"][index]["audio_bytes"] = audio_bytes

    def set_receipt_tip(self, index: int, tip_text: str, tip_language: str) -> None:
        self._state["receipt_history"][index]["data"]["savings_tip"] = tip_text
        self._state["receipt_history"][index]["data"]["tip_language"] = tip_language
        self.invalidate_rag_index()

    def get_editing_receipt_index(self) -> Optional[int]:
        return self._state.get("editing_receipt_index", None)

    def set_editing_receipt_index(self, value: Optional[int]) -> None:
        self._state["editing_receipt_index"] = value

    def clear_generated_insights(self) -> None:
        self._state["history_category_suggestions"] = []
        self._state["history_purchase_tips"] = []

    def get_history_category_suggestions(self) -> list[CategorySuggestion]:
        return list(self._state.get("history_category_suggestions", []))

    def set_history_category_suggestions(
        self,
        suggestions: list[CategorySuggestion],
    ) -> None:
        self._state["history_category_suggestions"] = suggestions

    def get_history_purchase_tips(self) -> list[HistoryTipEntry]:
        return list(self._state.get("history_purchase_tips", []))

    def set_history_purchase_tips(self, tips: list[HistoryTipEntry]) -> None:
        self._state["history_purchase_tips"] = tips

    def sync_category_budgets(self, categories: list[str]) -> None:
        current_budgets = self.get_category_budgets()
        self._state["category_budgets"] = {
            category: round(float(current_budgets.get(category, 0.0) or 0.0), 2)
            for category in categories
        }

    def get_category_budgets(self) -> dict[str, float]:
        return dict(self._state.get("category_budgets", {}))

    def set_category_budgets(self, budgets: dict[str, float]) -> None:
        self._state["category_budgets"] = dict(budgets)

    def get_selected_budget_month(self) -> str:
        return str(self._state.get("selected_budget_month", ""))

    def set_selected_budget_month(self, selected_month: str) -> None:
        self._state["selected_budget_month"] = selected_month

    def get_vector_store(self) -> Any:
        return self._state.get("vector_store", None)

    def set_vector_store(self, vector_store: Any) -> None:
        self._state["vector_store"] = vector_store

    def get_rag_receipt_count(self) -> int:
        return int(self._state.get("rag_receipt_count", 0))

    def set_rag_receipt_count(self, receipt_count: int) -> None:
        self._state["rag_receipt_count"] = int(receipt_count)

    def invalidate_rag_index(self) -> None:
        self._state["vector_store"] = None
        self._state["rag_receipt_count"] = 0

    def get_rag_chat_history(self) -> list[RagMessage]:
        return list(self._state.get("rag_chat_history", []))

    def append_rag_chat_message(self, role: str, content: str) -> None:
        self._state["rag_chat_history"].append({"role": role, "content": content})

    def clear_rag_chat_history(self) -> None:
        self._state["rag_chat_history"] = []

