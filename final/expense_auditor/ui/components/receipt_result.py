from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from expense_auditor.analytics.charts import build_receipt_category_figure
from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.container import AppServices
from expense_auditor.types import ReceiptEntry
from expense_auditor.ui.components.receipt_editor import render_receipt_editor


def render_receipt_result(
    index: int,
    entry: ReceiptEntry,
    categories: list[str],
    repo: SessionRepository,
    services: AppServices,
) -> None:
    data = entry["data"]
    currency = data.get("currency", "$")

    c1, c2, c3 = st.columns(3)
    c1.metric("\U0001f3ea Vendor", data.get("vendor", "Unknown"))
    c2.metric("\U0001f4c5 Date", data.get("date", "Unknown"))
    c3.metric("\U0001f4b5 Total", f"{currency} {data.get('total', 0):.2f}")

    st.markdown("**\U0001f4cb Itemised Expenses**")
    items = data.get("items", [])
    if items:
        df_items = pd.DataFrame(items)
        df_items.columns = ["Item", "Price", "Category"]
        df_items["Price"] = df_items["Price"].apply(lambda value: f"{currency} {value:.2f}")
        st.dataframe(df_items, use_container_width=True, hide_index=True)

    category_totals = data.get("category_totals", {})
    if category_totals:
        st.markdown("**\U0001f4ca Spending by Category**")
        fig = build_receipt_category_figure(category_totals, currency)
        st.plotly_chart(fig, use_container_width=True)

    tip = data.get("savings_tip", "")
    actions_col, tip_action_col = st.columns(2)
    is_editing = repo.get_editing_receipt_index() == index
    button_label = "Close editor" if is_editing else "Edit receipt"
    if actions_col.button(
        button_label,
        key=f"edit_receipt_button_{index}",
        use_container_width=True,
    ):
        repo.set_editing_receipt_index(None if is_editing else index)
        st.rerun()

    if not tip:
        if tip_action_col.button(
            "Generate tip based on receipt",
            key=f"generate_tip_button_{index}",
            use_container_width=True,
        ):
            with st.spinner("Generating savings tip..."):
                try:
                    tip_data = services.insights.generate_savings_tip(data)
                    tip_text = tip_data.get("savings_tip", "").strip()
                    tip_language = tip_data.get("tip_language", "es").strip() or "es"
                    if not tip_text:
                        st.error("The model did not return a savings tip. Please try again.")
                        return
                    audio_bytes, detected_lang = services.audio.generate_audio(tip_text, tip_language)
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected tip response. Please try again.")
                    return
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    return

            repo.set_receipt_tip(index, tip_text, detected_lang)
            repo.update_receipt_audio(index, audio_bytes)
            st.rerun()
    else:
        tip_action_col.empty()

    if is_editing:
        render_receipt_editor(index, entry, categories, repo)

    if tip:
        st.markdown("**\U0001f4a1 Savings Tip**")
        st.info(f"\U0001f3af {tip}")
        if entry.get("audio_bytes"):
            st.audio(entry["audio_bytes"], format="audio/mp3")
            st.caption("\U0001f50a Listen to your personalised savings tip")

