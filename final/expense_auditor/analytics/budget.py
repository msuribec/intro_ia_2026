from __future__ import annotations

import pandas as pd

from expense_auditor.parsers.receipts import parse_receipt_date
from expense_auditor.types import BudgetAnalytics, BudgetSource, ReceiptEntry


def build_budget_source_data(history: list[ReceiptEntry]) -> BudgetSource:
    item_rows = []
    receipt_rows = []
    skipped_receipts = 0

    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        parsed_date = parse_receipt_date(data.get("date", ""))
        if parsed_date is None:
            skipped_receipts += 1
            continue

        receipt_rows.append(
            {
                "receipt_index": index,
                "parsed_date": parsed_date,
                "vendor": data.get("vendor", "Unknown"),
                "currency": data.get("currency", "$"),
                "total": float(data.get("total", 0) or 0),
            }
        )

        for item in data.get("items", []):
            item_rows.append(
                {
                    "receipt_index": index,
                    "parsed_date": parsed_date,
                    "vendor": data.get("vendor", "Unknown"),
                    "currency": data.get("currency", "$"),
                    "item": item.get("name", ""),
                    "category": item.get("category", "Other"),
                    "price": float(item.get("price", 0) or 0),
                }
            )

    items_df = pd.DataFrame(
        item_rows,
        columns=[
            "receipt_index",
            "parsed_date",
            "vendor",
            "currency",
            "item",
            "category",
            "price",
        ],
    )
    receipts_df = pd.DataFrame(
        receipt_rows,
        columns=["receipt_index", "parsed_date", "vendor", "currency", "total"],
    )
    return {
        "items_df": items_df,
        "receipts_df": receipts_df,
        "skipped_receipts": skipped_receipts,
    }


def get_default_budget_month(receipts_df: pd.DataFrame) -> str:
    if not receipts_df.empty:
        latest_date = receipts_df["parsed_date"].max()
        return latest_date.strftime("%Y-%m")
    return pd.Timestamp.today().strftime("%Y-%m")


def normalize_selected_budget_month(
    receipts_df: pd.DataFrame,
    selected_budget_month: str,
) -> str:
    if pd.notna(selected_budget_month) and isinstance(selected_budget_month, str):
        if pd.Series([selected_budget_month]).str.fullmatch(r"\d{4}-\d{2}").iloc[0]:
            return selected_budget_month
    return get_default_budget_month(receipts_df)


def build_month_budget_analytics(
    budget_source: BudgetSource,
    category_budgets: dict[str, float],
    selected_month_id: str,
) -> BudgetAnalytics:
    receipts_df = budget_source["receipts_df"]
    items_df = budget_source["items_df"]
    month_period = pd.Period(selected_month_id, freq="M")

    budget_series = pd.Series(category_budgets, dtype="float64")
    if budget_series.empty:
        budget_series = pd.Series(dtype="float64")

    if not receipts_df.empty:
        month_receipts_df = receipts_df[
            receipts_df["parsed_date"].dt.to_period("M") == month_period
        ].copy()
    else:
        month_receipts_df = receipts_df.copy()

    if not items_df.empty:
        month_items_df = items_df[
            items_df["parsed_date"].dt.to_period("M") == month_period
        ].copy()
    else:
        month_items_df = items_df.copy()

    actual_series = (
        month_items_df.groupby("category")["price"].sum()
        if not month_items_df.empty
        else pd.Series(dtype="float64")
    )
    all_categories = list(
        dict.fromkeys(budget_series.index.tolist() + actual_series.index.tolist())
    )
    budget_series = budget_series.reindex(all_categories, fill_value=0.0)
    actual_series = actual_series.reindex(all_categories, fill_value=0.0)

    category_df = pd.DataFrame(
        {
            "Category": budget_series.index.tolist(),
            "Budget": budget_series.values,
            "Actual": actual_series.values,
        }
    )
    if category_df.empty:
        category_df = pd.DataFrame(columns=["Category", "Budget", "Actual"])
    category_df["Variance"] = category_df["Actual"] - category_df["Budget"]
    category_df["Variance Status"] = category_df["Variance"].apply(
        lambda value: "Overspent" if value > 0 else "Within budget"
    )

    stacked_df = category_df.melt(
        id_vars="Category",
        value_vars=["Budget", "Actual"],
        var_name="Scenario",
        value_name="Amount",
    )

    daily_index = pd.date_range(
        month_period.start_time.normalize(),
        month_period.end_time.normalize(),
        freq="D",
    )
    if not month_items_df.empty:
        daily_actual_series = month_items_df.groupby("parsed_date")["price"].sum()
        daily_actual_series = daily_actual_series.reindex(daily_index, fill_value=0.0)
    else:
        daily_actual_series = pd.Series(0.0, index=daily_index)

    daily_df = pd.DataFrame(
        {
            "Date": daily_index,
            "Actual Daily": daily_actual_series.values,
        }
    )
    daily_df["Day"] = daily_df["Date"].dt.day

    total_budget = float(category_df["Budget"].sum()) if not category_df.empty else 0.0
    total_actual = float(category_df["Actual"].sum()) if not category_df.empty else 0.0
    days_in_month = len(daily_df) if not daily_df.empty else month_period.days_in_month
    daily_df["Ideal Cumulative"] = (
        total_budget * daily_df["Day"] / max(days_in_month, 1)
        if not daily_df.empty
        else 0.0
    )
    daily_df["Actual Cumulative"] = daily_df["Actual Daily"].cumsum()

    currency = "$"
    if not month_items_df.empty:
        currency = str(month_items_df["currency"].dropna().iloc[-1])
    elif not month_receipts_df.empty:
        currency = str(month_receipts_df["currency"].dropna().iloc[-1])

    return {
        "month_period": month_period,
        "month_receipts_df": month_receipts_df,
        "month_items_df": month_items_df,
        "category_df": category_df,
        "stacked_df": stacked_df,
        "daily_df": daily_df,
        "total_budget": round(total_budget, 2),
        "total_actual": round(total_actual, 2),
        "currency": currency,
    }

