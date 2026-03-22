from __future__ import annotations

import streamlit as st

from expense_auditor.analytics.charts import (
    build_dashboard_category_pie_figure,
    build_dashboard_stack_figure,
    build_dashboard_totals_figure,
)
from expense_auditor.analytics.history import build_dashboard_summary
from expense_auditor.persistence.session_repository import SessionRepository


def render_dashboard_tab(repo: SessionRepository) -> None:
    history = repo.get_receipt_history()
    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
        return

    summary = build_dashboard_summary(history)
    k1, k2, k3 = st.columns(3)
    k1.metric("\U0001f9fe Receipts analyzed", len(history))
    k2.metric("\U0001f4b0 Grand total", f"{summary['currency']} {summary['grand_total']:.2f}")
    k3.metric("\U0001f4cc Top category", summary["top_category"])

    st.divider()

    combined_categories = summary["combined_categories"]
    if combined_categories:
        st.subheader("Spending by Category (all receipts)")
        st.plotly_chart(
            build_dashboard_category_pie_figure(combined_categories),
            use_container_width=True,
        )

    st.divider()

    st.subheader("Total per Receipt")
    st.plotly_chart(
        build_dashboard_totals_figure(
            summary["receipt_labels"],
            summary["receipt_totals"],
            summary["currency"],
        ),
        use_container_width=True,
    )

    st.divider()

    st.subheader("Category Breakdown per Receipt")
    if summary["receipt_category_rows"]:
        st.plotly_chart(
            build_dashboard_stack_figure(
                summary["receipt_category_rows"],
                summary["currency"],
            ),
            use_container_width=True,
        )

