from __future__ import annotations

import pandas as pd

from expense_auditor.constants import SPENDING_EXPORT_COLUMNS
from expense_auditor.types import ReceiptEntry


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_categories_export_csv(categories: list[str]) -> bytes:
    return dataframe_to_csv_bytes(pd.DataFrame({"category": categories}))


def build_spending_export_csv(history: list[ReceiptEntry]) -> bytes:
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        vendor = data.get("vendor", "Unknown")
        date = data.get("date", "Unknown")
        currency = data.get("currency", "$")
        receipt_total = float(data.get("total", 0) or 0)

        for item in data.get("items", []):
            rows.append(
                {
                    "receipt_index": index,
                    "vendor": vendor,
                    "date": date,
                    "currency": currency,
                    "item": item.get("name", ""),
                    "price": float(item.get("price", 0) or 0),
                    "category": item.get("category", ""),
                    "receipt_total": receipt_total,
                }
            )

    df = pd.DataFrame(rows, columns=SPENDING_EXPORT_COLUMNS)
    return dataframe_to_csv_bytes(df)

