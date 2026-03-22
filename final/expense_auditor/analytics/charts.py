from __future__ import annotations

import calendar

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_receipt_category_figure(category_totals: dict[str, float], currency: str) -> go.Figure:
    df_cat = pd.DataFrame(
        list(category_totals.items()),
        columns=["Category", "Amount"],
    ).sort_values("Amount", ascending=False)
    fig = px.bar(
        df_cat,
        x="Category",
        y="Amount",
        labels={"Amount": f"Amount ({currency})"},
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    return fig


def build_dashboard_category_pie_figure(combined_categories: dict[str, float]) -> go.Figure:
    df_pie = pd.DataFrame(list(combined_categories.items()), columns=["Category", "Amount"])
    fig = px.pie(
        df_pie,
        names="Category",
        values="Amount",
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True, margin=dict(t=30, b=10))
    return fig


def build_dashboard_totals_figure(
    receipt_labels: list[str],
    receipt_totals: list[float],
    currency: str,
) -> go.Figure:
    df_totals = pd.DataFrame({"Receipt": receipt_labels, "Total": receipt_totals})
    fig = px.bar(
        df_totals,
        x="Receipt",
        y="Total",
        labels={"Total": f"Total ({currency})"},
        color="Receipt",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text_auto=".2f",
    )
    fig.update_layout(showlegend=False, margin=dict(t=20, b=60))
    fig.update_xaxes(tickangle=-25)
    return fig


def build_dashboard_stack_figure(rows: list[dict[str, object]], currency: str) -> go.Figure:
    df_stack = pd.DataFrame(rows)
    fig = px.bar(
        df_stack,
        x="Receipt",
        y="Amount",
        color="Category",
        labels={"Amount": f"Amount ({currency})"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        barmode="stack",
    )
    fig.update_layout(margin=dict(t=20, b=60))
    fig.update_xaxes(tickangle=-25)
    return fig


def build_budget_vs_actual_figure(
    budget_vs_actual_df: pd.DataFrame,
    currency: str,
) -> go.Figure:
    fig = px.bar(
        budget_vs_actual_df,
        x="Category",
        y="Amount",
        color="Type",
        barmode="group",
        labels={"Amount": f"Amount ({currency})"},
        color_discrete_map={"Budget": "#94a3b8", "Actual": "#16a34a"},
    )
    fig.update_layout(margin=dict(t=20, b=60))
    fig.update_xaxes(tickangle=-25)
    return fig


def build_budget_stacked_figure(stacked_df: pd.DataFrame, currency: str) -> go.Figure:
    fig = px.bar(
        stacked_df,
        x="Scenario",
        y="Amount",
        color="Category",
        labels={"Amount": f"Amount ({currency})"},
        color_discrete_sequence=px.colors.qualitative.Safe,
        barmode="stack",
    )
    fig.update_layout(margin=dict(t=20, b=20))
    return fig


def build_budget_line_figure(line_df: pd.DataFrame, currency: str) -> go.Figure:
    fig = px.line(
        line_df,
        x="Date",
        y="Amount",
        color="Series",
        labels={"Amount": f"Amount ({currency})"},
        color_discrete_map={
            "Ideal Cumulative": "#94a3b8",
            "Actual Cumulative": "#2563eb",
        },
    )
    fig.update_layout(margin=dict(t=20, b=20))
    return fig


def build_budget_actual_pie_figure(pie_df: pd.DataFrame) -> go.Figure:
    fig = px.pie(
        pie_df,
        names="Category",
        values="Actual",
        color_discrete_sequence=px.colors.qualitative.Safe,
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(margin=dict(t=20, b=20))
    return fig


def build_budget_variance_figure(category_df: pd.DataFrame, currency: str) -> go.Figure:
    fig = px.bar(
        category_df,
        x="Category",
        y="Variance",
        color="Variance Status",
        labels={"Variance": f"Variance ({currency})"},
        color_discrete_map={
            "Overspent": "#dc2626",
            "Within budget": "#16a34a",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="#475569")
    fig.update_layout(margin=dict(t=20, b=60), showlegend=False)
    fig.update_xaxes(tickangle=-25)
    return fig


def build_budget_gauge_figure(usage_pct: float, total_budget: float) -> go.Figure:
    gauge_max = max(100.0, usage_pct + 10.0) if total_budget > 0 else 100.0
    gauge_title = "Total spending vs budget"
    if total_budget == 0:
        gauge_title = "Set a budget to track total usage"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=usage_pct if total_budget > 0 else 0.0,
            number={"suffix": "%"},
            title={"text": gauge_title},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, min(100.0, gauge_max)], "color": "#dbeafe"},
                    {"range": [100.0, gauge_max], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#dc2626", "width": 4},
                    "thickness": 0.75,
                    "value": 100.0,
                },
            },
        )
    )
    fig.update_layout(margin=dict(t=40, b=20))
    return fig


def build_budget_heatmap_figure(
    month_period: pd.Period,
    daily_df: pd.DataFrame,
    currency: str,
) -> go.Figure:
    spend_by_day = {
        int(row["Day"]): float(row["Actual Daily"])
        for _, row in daily_df.iterrows()
    }
    month_matrix = calendar.Calendar(firstweekday=0).monthdayscalendar(
        month_period.year,
        month_period.month,
    )

    z_values = []
    text_values = []
    for week in month_matrix:
        week_values = []
        week_text = []
        for day in week:
            if day == 0:
                week_values.append(None)
                week_text.append("")
            else:
                amount = spend_by_day.get(day, 0.0)
                week_values.append(amount)
                week_text.append(f"Day {day}<br>{currency} {amount:.2f}")
        z_values.append(week_values)
        text_values.append(week_text)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            y=[f"Week {index + 1}" for index in range(len(month_matrix))],
            text=text_values,
            hovertemplate="%{text}<extra></extra>",
            colorscale="YlGnBu",
            colorbar={"title": f"Spend ({currency})"},
            zmin=0,
        )
    )
    fig.update_layout(
        margin=dict(t=30, b=10),
        xaxis_title="Day of week",
        yaxis_title="Week of month",
    )
    fig.update_yaxes(autorange="reversed")
    return fig

