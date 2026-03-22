from __future__ import annotations

import streamlit as st

from expense_auditor.constants import PAGE_ICON, PAGE_LAYOUT, PAGE_TITLE


def configure_page() -> None:
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=PAGE_ICON,
        layout=PAGE_LAYOUT,
    )
    st.title("\U0001f4b0 My-Expense Auditor")
    st.markdown(
        "*Upload a receipt and your expense categories — extract expenses, edit them, and generate a savings tip by voice.*"
    )
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] button[kind="primary"] {
            background-color: #16a34a; border-color: #16a34a; color: white;
        }
        div[data-testid="stButton"] button[kind="primary"]:hover {
            background-color: #15803d; border-color: #15803d; color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

