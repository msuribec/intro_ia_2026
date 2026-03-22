from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from expense_auditor.constants import SPENDING_EXPORT_COLUMNS
from expense_auditor.types import ReceiptData, ReceiptEntry, ReceiptItem


def rebuild_receipt_data(
    original_data: ReceiptData | dict,
    vendor: str,
    date: str,
    currency: str,
    items: list[ReceiptItem],
) -> ReceiptData:
    category_totals: dict[str, float] = {}
    total = 0.0

    for item in items:
        price = round(float(item["price"]), 2)
        category = item["category"]
        total += price
        category_totals[category] = round(category_totals.get(category, 0.0) + price, 2)

    return {
        **original_data,
        "vendor": vendor.strip() or "Unknown",
        "date": date.strip() or "Unknown",
        "currency": currency.strip() or "$",
        "items": items,
        "total": round(total, 2),
        "category_totals": category_totals,
        "savings_tip": "",
        "tip_language": "",
    }


def parse_uploaded_spending_history(file) -> list[ReceiptEntry]:
    content = file.read().decode("utf-8")
    file.seek(0)

    try:
        df = pd.read_csv(io.StringIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc

    missing_columns = [column for column in SPENDING_EXPORT_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))

    df = df[SPENDING_EXPORT_COLUMNS].copy()
    for column in ["vendor", "date", "currency", "item", "category"]:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["vendor"] = df["vendor"].replace("", "Unknown")
    df["date"] = df["date"].replace("", "Unknown")
    df["currency"] = df["currency"].replace("", "$")

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["receipt_total"] = pd.to_numeric(df["receipt_total"], errors="coerce")

    invalid_price_rows = df.index[df["price"].isna()].tolist()
    if invalid_price_rows:
        rows_text = ", ".join(str(row + 2) for row in invalid_price_rows)
        raise ValueError(f"Invalid price values found in CSV rows: {rows_text}")

    blank_item_rows = df.index[df["item"] == ""].tolist()
    if blank_item_rows:
        rows_text = ", ".join(str(row + 2) for row in blank_item_rows)
        raise ValueError(f"Blank item values found in CSV rows: {rows_text}")

    blank_category_rows = df.index[df["category"] == ""].tolist()
    if blank_category_rows:
        rows_text = ", ".join(str(row + 2) for row in blank_category_rows)
        raise ValueError(f"Blank category values found in CSV rows: {rows_text}")

    imported_entries: list[ReceiptEntry] = []
    for _, receipt_df in df.groupby("receipt_index", sort=False):
        first_row = receipt_df.iloc[0]
        items = [
            {
                "name": row["item"],
                "price": round(float(row["price"]), 2),
                "category": row["category"],
            }
            for _, row in receipt_df.iterrows()
        ]
        data = rebuild_receipt_data(
            {},
            str(first_row["vendor"]),
            str(first_row["date"]),
            str(first_row["currency"]),
            items,
        )
        imported_entries.append(
            {
                "image_bytes": None,
                "source": "imported_csv",
                "data": data,
                "audio_bytes": None,
            }
        )

    if not imported_entries:
        raise ValueError("The uploaded CSV does not contain any receipt rows.")

    return imported_entries


def parse_receipt_date(date_value: str) -> Optional[pd.Timestamp]:
    if date_value is None:
        return None

    text = str(date_value).strip()
    if not text or text.lower() in {"unknown", "none", "null", "nan", "nat"}:
        return None

    attempts = [
        {"format": "mixed", "dayfirst": False},
        {"format": "mixed", "dayfirst": True},
        {"dayfirst": False},
        {"dayfirst": True},
    ]
    for kwargs in attempts:
        try:
            parsed = pd.to_datetime(text, errors="coerce", **kwargs)
        except (TypeError, ValueError):
            continue
        if pd.notna(parsed):
            return pd.Timestamp(parsed).normalize()
    return None

