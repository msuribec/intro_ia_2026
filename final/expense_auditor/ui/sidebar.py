from __future__ import annotations

from typing import Any

import streamlit as st

from expense_auditor.parsers.receipts import parse_uploaded_spending_history
from expense_auditor.persistence.csv_io import (
    build_categories_export_csv,
    build_spending_export_csv,
)
from expense_auditor.persistence.session_repository import SessionRepository


def render_sidebar(repo: SessionRepository) -> tuple[str, Any]:
    widget_seed = repo.get_widget_seed()

    with st.sidebar:
        st.header("\u2699\ufe0f Configuration")
        st.header("Set up API Key \U0001f5dd\ufe0f")
        api_key = st.text_input(
            "Paste your Google Gemini API Key",
            key=f"api_key_{widget_seed}",
            type="password",
            placeholder="AIza...",
            help=(
                "Free at aistudio.google.com — no credit card needed.\n\n"
                "How to get a free key:\n"
                "1. Go to Google AI Studio: https://aistudio.google.com\n"
                "2. Sign in with Google\n"
                "3. Click Get API Key → Create API Key\n"
                "4. Paste it above"
            ),
        )

        categories_file = None
        if api_key:
            st.markdown("---")
            st.header("Load your categories file \U0001f4c2")
            categories_file = st.file_uploader(
                "Your categories file (TXT or CSV)",
                type=["csv", "txt"],
                key=f"categories_file_{widget_seed}",
            )
        else:
            st.caption("Enter your API key to unlock category upload.")

        st.markdown("---")
        st.header("Export data")
        approved_categories = repo.get_approved_categories()
        categories_csv = build_categories_export_csv(approved_categories)
        st.download_button(
            "Download categories CSV",
            data=categories_csv,
            file_name="expense_categories.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not repo.get_categories_approved(),
        )
        if not repo.get_categories_approved():
            st.caption("Approve categories to enable category export.")

        spending_history = repo.get_receipt_history()
        spending_csv = build_spending_export_csv(spending_history) if spending_history else b""
        st.download_button(
            "Download spending CSV",
            data=spending_csv,
            file_name="spending_data.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=not spending_history,
        )
        if not spending_history:
            st.caption("Analyze at least one receipt to export spending data.")

        st.markdown("---")
        st.header("Upload receipt history")
        receipt_history_file = st.file_uploader(
            "Import a spending history CSV",
            type=["csv"],
            key=f"receipt_history_file_{widget_seed}",
            disabled=not repo.get_categories_approved(),
        )
        import_history_clicked = st.button(
            "Import receipt history",
            use_container_width=True,
            disabled=not (repo.get_categories_approved() and receipt_history_file),
        )
        if not repo.get_categories_approved():
            st.caption("Approve categories to enable receipt history import.")
        elif import_history_clicked and receipt_history_file is not None:
            try:
                imported_entries = parse_uploaded_spending_history(receipt_history_file)
            except ValueError as exc:
                st.error(str(exc))
            else:
                repo.extend_receipt_history(imported_entries)
                repo.clear_generated_insights()
                st.success(f"Imported {len(imported_entries)} receipt(s) from CSV.")

        st.markdown("---")
        if st.button("End session", use_container_width=True):
            repo.restart_streamlit_app()

    return api_key, categories_file

