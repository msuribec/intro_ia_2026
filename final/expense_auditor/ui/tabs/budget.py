from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from expense_auditor.analytics.budget import (
    build_budget_source_data,
    build_month_budget_analytics,
    normalize_selected_budget_month,
)
from expense_auditor.analytics.charts import (
    build_budget_actual_pie_figure,
    build_budget_gauge_figure,
    build_budget_heatmap_figure,
    build_budget_line_figure,
    build_budget_stacked_figure,
    build_budget_variance_figure,
    build_budget_vs_actual_figure,
)
from expense_auditor.persistence.session_repository import SessionRepository


def render_budget_tab(repo: SessionRepository, categories: list[str]) -> None:
    history = repo.get_receipt_history()
    budget_source = build_budget_source_data(history)
    receipts_df = budget_source["receipts_df"]

    selected_budget_month = normalize_selected_budget_month(
        receipts_df,
        repo.get_selected_budget_month(),
    )
    repo.set_selected_budget_month(selected_budget_month)

    st.subheader("Monthly budget vs actual")
    st.caption("Set one shared monthly budget per category and compare it against any month of receipt history.")

    selected_year = int(selected_budget_month[:4])
    selected_month_number = int(selected_budget_month[5:7])
    parseable_years = (
        {int(value.year) for value in receipts_df["parsed_date"]}
        if not receipts_df.empty
        else set()
    )
    year_options = sorted(
        parseable_years | {pd.Timestamp.today().year, selected_year},
        reverse=True,
    )

    picker_col, month_col = st.columns(2)
    selected_year = picker_col.selectbox(
        "Year",
        options=year_options,
        index=year_options.index(selected_year),
        key="budget_year_picker",
    )
    selected_month_number = month_col.selectbox(
        "Month",
        options=list(range(1, 13)),
        index=selected_month_number - 1,
        format_func=lambda month: calendar.month_name[month],
        key="budget_month_picker",
    )
    selected_budget_month = f"{selected_year}-{selected_month_number:02d}"
    repo.set_selected_budget_month(selected_budget_month)

    st.markdown("#### Shared monthly budget by category")
    st.caption("These amounts are reused for every month you select in this tab.")

    category_budgets = repo.get_category_budgets()
    with st.form("budget_amounts_form"):
        budget_inputs: dict[str, float] = {}
        columns = st.columns(2)
        for index, category in enumerate(categories):
            budget_inputs[category] = columns[index % 2].number_input(
                category,
                min_value=0.0,
                value=float(category_budgets.get(category, 0.0)),
                step=1.0,
                format="%.2f",
                key=f"budget_input_{category}",
            )
        budget_saved = st.form_submit_button(
            "Save budget amounts",
            type="primary",
            use_container_width=True,
        )

    if budget_saved:
        category_budgets = {
            category: round(float(amount), 2)
            for category, amount in budget_inputs.items()
        }
        repo.set_category_budgets(category_budgets)
    else:
        category_budgets = repo.get_category_budgets()

    analytics = build_month_budget_analytics(
        budget_source,
        category_budgets,
        selected_budget_month,
    )
    month_label = analytics["month_period"].start_time.strftime("%B %Y")
    usage_pct = (
        (analytics["total_actual"] / analytics["total_budget"]) * 100
        if analytics["total_budget"] > 0
        else 0.0
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("\U0001f4bc Total budget", f"{analytics['currency']} {analytics['total_budget']:.2f}")
    k2.metric("\U0001f4b8 Actual spend", f"{analytics['currency']} {analytics['total_actual']:.2f}")
    k3.metric("\U0001f4c8 Budget used", f"{usage_pct:.1f}%")

    if budget_source["skipped_receipts"] > 0:
        st.warning(
            f"{budget_source['skipped_receipts']} receipt(s) were excluded from budget charts because their dates were missing or could not be parsed."
        )

    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
        return

    if receipts_df.empty:
        st.info("No dated receipts are available for budget analysis yet. Upload or edit receipts with recognizable dates to unlock the charts.")
        return

    if analytics["month_receipts_df"].empty:
        st.info(f"No dated receipts are available for {month_label}. Pick another month or add receipts with dates in that month.")
        return

    category_df = analytics["category_df"].copy()
    budget_vs_actual_df = category_df.melt(
        id_vars="Category",
        value_vars=["Budget", "Actual"],
        var_name="Type",
        value_name="Amount",
    )

    st.divider()
    st.subheader(f"Budget analysis for {month_label}")
    st.plotly_chart(
        build_budget_vs_actual_figure(budget_vs_actual_df, analytics["currency"]),
        use_container_width=True,
    )

    st.divider()
    st.plotly_chart(
        build_budget_stacked_figure(analytics["stacked_df"], analytics["currency"]),
        use_container_width=True,
    )

    st.divider()
    line_df = analytics["daily_df"].melt(
        id_vars=["Date"],
        value_vars=["Ideal Cumulative", "Actual Cumulative"],
        var_name="Series",
        value_name="Amount",
    )
    st.plotly_chart(
        build_budget_line_figure(line_df, analytics["currency"]),
        use_container_width=True,
    )

    st.divider()
    pie_df = category_df[category_df["Actual"] > 0][["Category", "Actual"]]
    if not pie_df.empty:
        st.plotly_chart(
            build_budget_actual_pie_figure(pie_df),
            use_container_width=True,
        )
    else:
        st.info("No actual spending was recorded in this month, so the spending breakdown pie chart is empty.")

    st.divider()
    st.plotly_chart(
        build_budget_variance_figure(category_df, analytics["currency"]),
        use_container_width=True,
    )

    st.divider()
    st.plotly_chart(
        build_budget_gauge_figure(usage_pct, analytics["total_budget"]),
        use_container_width=True,
    )

    st.divider()
    st.plotly_chart(
        build_budget_heatmap_figure(
            analytics["month_period"],
            analytics["daily_df"],
            analytics["currency"],
        ),
        use_container_width=True,
    )

