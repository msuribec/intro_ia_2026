from __future__ import annotations

import pandas as pd

from expense_auditor.types import HistorySummary, ReceiptEntry


def build_history_summary(history: list[ReceiptEntry]) -> HistorySummary:
    receipts = []
    stores = []
    items = []
    category_totals: dict[str, float] = {}

    for index, entry in enumerate(history, start=1):
        data = entry.get("data", {})
        vendor = data.get("vendor", "Unknown")
        stores.append(vendor)
        receipt_items = data.get("items", [])

        receipts.append(
            {
                "receipt_index": index,
                "vendor": vendor,
                "date": data.get("date", "Unknown"),
                "currency": data.get("currency", "$"),
                "total": float(data.get("total", 0) or 0),
                "items_count": len(receipt_items),
            }
        )

        for item in receipt_items:
            item_name = item.get("name", "")
            item_category = item.get("category", "")
            item_price = float(item.get("price", 0) or 0)
            items.append(
                {
                    "receipt_index": index,
                    "vendor": vendor,
                    "item": item_name,
                    "category": item_category,
                    "price": item_price,
                }
            )
            category_totals[item_category] = round(
                category_totals.get(item_category, 0.0) + item_price,
                2,
            )

    top_stores = pd.Series(stores).value_counts().head(10).to_dict() if stores else {}
    return {
        "receipts": receipts,
        "items": items,
        "top_stores": top_stores,
        "category_totals": category_totals,
    }


def build_dashboard_summary(history: list[ReceiptEntry]) -> dict[str, object]:
    if not history:
        return {
            "currency": "$",
            "grand_total": 0.0,
            "top_category": "\u2014",
            "combined_categories": {},
            "receipt_labels": [],
            "receipt_totals": [],
            "receipt_category_rows": [],
        }

    currency = history[-1]["data"].get("currency", "$")
    grand_total = sum(float(entry["data"].get("total", 0) or 0) for entry in history)
    combined_categories: dict[str, float] = {}
    for entry in history:
        for category, amount in entry["data"].get("category_totals", {}).items():
            combined_categories[category] = combined_categories.get(category, 0.0) + amount

    top_category = max(combined_categories, key=combined_categories.get) if combined_categories else "\u2014"
    receipt_labels = [
        f"{entry['data'].get('vendor', 'Unknown')} ({entry['data'].get('date', '?')})"
        for entry in history
    ]
    receipt_totals = [float(entry["data"].get("total", 0) or 0) for entry in history]

    rows = []
    for label, entry in zip(receipt_labels, history):
        for category, amount in entry["data"].get("category_totals", {}).items():
            rows.append({"Receipt": label, "Category": category, "Amount": amount})

    return {
        "currency": currency,
        "grand_total": round(grand_total, 2),
        "top_category": top_category,
        "combined_categories": combined_categories,
        "receipt_labels": receipt_labels,
        "receipt_totals": receipt_totals,
        "receipt_category_rows": rows,
    }

