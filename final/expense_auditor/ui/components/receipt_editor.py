from __future__ import annotations

import pandas as pd
import streamlit as st

from expense_auditor.parsers.receipts import rebuild_receipt_data
from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.types import ReceiptEntry
from expense_auditor.validators.receipts import normalize_edited_items


def render_receipt_editor(
    index: int,
    entry: ReceiptEntry,
    categories: list[str],
    repo: SessionRepository,
) -> None:
    data = entry["data"]
    editable_categories = list(
        dict.fromkeys(
            categories
            + [
                str(item.get("category", "")).strip()
                for item in data.get("items", [])
                if str(item.get("category", "")).strip()
            ]
        )
    )
    items_df = pd.DataFrame(data.get("items", []), columns=["name", "price", "category"])

    with st.expander("Edit receipt details", expanded=True):
        with st.form(f"edit_receipt_form_{index}"):
            vendor = st.text_input("Vendor", value=data.get("vendor", ""))
            date = st.text_input("Date", value=data.get("date", ""))
            currency = st.text_input("Currency", value=data.get("currency", "$"))
            edited_items_df = st.data_editor(
                items_df,
                key=f"receipt_items_editor_{index}",
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                column_config={
                    "name": st.column_config.TextColumn("Item"),
                    "price": st.column_config.NumberColumn(
                        "Price",
                        min_value=0.0,
                        step=0.01,
                        format="%.2f",
                    ),
                    "category": st.column_config.SelectboxColumn(
                        "Category",
                        options=editable_categories,
                    ),
                },
            )

            save_col, cancel_col = st.columns(2)
            save_clicked = save_col.form_submit_button(
                "Save changes",
                type="primary",
                use_container_width=True,
            )
            cancel_clicked = cancel_col.form_submit_button(
                "Cancel",
                use_container_width=True,
            )

        if cancel_clicked:
            repo.set_editing_receipt_index(None)
            st.rerun()

        if save_clicked:
            cleaned_items, errors = normalize_edited_items(edited_items_df, editable_categories)
            if errors:
                for error in errors:
                    st.error(error)
                return

            updated_data = rebuild_receipt_data(data, vendor, date, currency, cleaned_items)
            repo.update_receipt_data(index, updated_data)
            repo.update_receipt_audio(index, None)
            repo.set_editing_receipt_index(None)
            repo.clear_generated_insights()
            st.rerun()

