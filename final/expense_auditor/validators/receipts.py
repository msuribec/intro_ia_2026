from __future__ import annotations

import pandas as pd

from expense_auditor.types import ReceiptItem


def normalize_edited_items(
    items_df: pd.DataFrame,
    categories: list[str],
) -> tuple[list[ReceiptItem], list[str]]:
    cleaned_items: list[ReceiptItem] = []
    errors: list[str] = []

    for row_number, row in enumerate(items_df.to_dict("records"), start=1):
        raw_name = row.get("name", "")
        raw_category = row.get("category", "")
        raw_price = row.get("price", None)

        name = "" if pd.isna(raw_name) else str(raw_name).strip()
        category = "" if pd.isna(raw_category) else str(raw_category).strip()
        price_missing = raw_price is None or pd.isna(raw_price)

        if not name and not category and price_missing:
            continue

        if not name:
            errors.append(f"Row {row_number}: item name is required.")

        if category not in categories:
            errors.append(f"Row {row_number}: category must be one of the approved categories.")

        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            errors.append(f"Row {row_number}: price must be a valid number.")
            continue

        if price < 0:
            errors.append(f"Row {row_number}: price cannot be negative.")

        if name and category in categories and price >= 0:
            cleaned_items.append(
                {
                    "name": name,
                    "price": round(price, 2),
                    "category": category,
                }
            )

    if not cleaned_items:
        errors.append("Add at least one valid item before saving.")

    return cleaned_items, errors

