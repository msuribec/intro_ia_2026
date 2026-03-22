from __future__ import annotations

import io
import unittest

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

from expense_auditor.parsers.gemini_json import parse_model_json_response

if pd is not None:
    from expense_auditor.parsers.categories import parse_categories
    from expense_auditor.parsers.receipts import parse_receipt_date, parse_uploaded_spending_history
    from expense_auditor.persistence.csv_io import build_spending_export_csv
    from expense_auditor.validators.receipts import normalize_edited_items


class NamedBytesIO(io.BytesIO):
    def __init__(self, content: str, name: str) -> None:
        super().__init__(content.encode("utf-8"))
        self.name = name


@unittest.skipIf(pd is None, "pandas is not installed")
class ParserAndValidatorTests(unittest.TestCase):
    def test_parse_categories_from_text(self) -> None:
        file = NamedBytesIO("Food\nTransport\n", "categories.txt")
        self.assertEqual(parse_categories(file), ["Food", "Transport"])

    def test_parse_categories_from_csv_uses_detected_column(self) -> None:
        file = NamedBytesIO("categoria\nFood\nTransport\n", "categories.csv")
        self.assertEqual(parse_categories(file), ["Food", "Transport"])

    def test_parse_model_json_response_strips_markdown_fences(self) -> None:
        payload = "```json\n{\"vendor\": \"Store\"}\n```"
        self.assertEqual(parse_model_json_response(payload), {"vendor": "Store"})

    def test_normalize_edited_items_collects_errors(self) -> None:
        items_df = pd.DataFrame(
            [
                {"name": "Milk", "price": 3.25, "category": "Food"},
                {"name": "", "price": 1.0, "category": "Food"},
                {"name": "Taxi", "price": -2.0, "category": "Transport"},
            ]
        )
        cleaned_items, errors = normalize_edited_items(items_df, ["Food", "Transport"])

        self.assertEqual(cleaned_items, [{"name": "Milk", "price": 3.25, "category": "Food"}])
        self.assertTrue(any("item name is required" in error for error in errors))
        self.assertTrue(any("price cannot be negative" in error for error in errors))

    def test_parse_uploaded_spending_history_rejects_missing_columns(self) -> None:
        file = NamedBytesIO("vendor,date\nStore,2025-01-01\n", "history.csv")
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            parse_uploaded_spending_history(file)

    def test_spending_csv_roundtrip_preserves_receipt_data(self) -> None:
        history = [
            {
                "image_bytes": None,
                "data": {
                    "vendor": "Market",
                    "date": "2025-03-01",
                    "currency": "$",
                    "items": [
                        {"name": "Milk", "price": 3.5, "category": "Food"},
                        {"name": "Bread", "price": 2.0, "category": "Food"},
                    ],
                    "total": 5.5,
                    "category_totals": {"Food": 5.5},
                    "savings_tip": "",
                    "tip_language": "",
                },
                "audio_bytes": None,
            }
        ]

        csv_bytes = build_spending_export_csv(history)
        imported = parse_uploaded_spending_history(NamedBytesIO(csv_bytes.decode("utf-8"), "history.csv"))

        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0]["data"]["vendor"], "Market")
        self.assertEqual(imported[0]["data"]["total"], 5.5)
        self.assertEqual(imported[0]["data"]["category_totals"], {"Food": 5.5})

    def test_parse_receipt_date_handles_unknown_values(self) -> None:
        self.assertIsNone(parse_receipt_date("Unknown"))
        self.assertEqual(str(parse_receipt_date("2025-03-01").date()), "2025-03-01")


if __name__ == "__main__":
    unittest.main()
