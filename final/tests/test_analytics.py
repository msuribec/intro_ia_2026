from __future__ import annotations

import unittest

try:
    import pandas  # noqa: F401
except ModuleNotFoundError:
    pandas = None

if pandas is not None:
    from expense_auditor.analytics.budget import (
        build_budget_source_data,
        build_month_budget_analytics,
        normalize_selected_budget_month,
    )
    from expense_auditor.analytics.history import build_dashboard_summary, build_history_summary


def sample_history():
    return [
        {
            "image_bytes": None,
            "data": {
                "vendor": "Market",
                "date": "2025-03-01",
                "currency": "$",
                "items": [
                    {"name": "Milk", "price": 4.0, "category": "Food"},
                    {"name": "Soap", "price": 6.0, "category": "Household"},
                ],
                "total": 10.0,
                "category_totals": {"Food": 4.0, "Household": 6.0},
                "savings_tip": "",
                "tip_language": "",
            },
            "audio_bytes": None,
        },
        {
            "image_bytes": None,
            "data": {
                "vendor": "Bus",
                "date": "Unknown",
                "currency": "$",
                "items": [
                    {"name": "Ticket", "price": 3.0, "category": "Transport"},
                ],
                "total": 3.0,
                "category_totals": {"Transport": 3.0},
                "savings_tip": "",
                "tip_language": "",
            },
            "audio_bytes": None,
        },
    ]


@unittest.skipIf(pandas is None, "pandas is not installed")
class AnalyticsTests(unittest.TestCase):
    def test_build_history_summary_aggregates_categories(self) -> None:
        summary = build_history_summary(sample_history())
        self.assertEqual(summary["category_totals"]["Food"], 4.0)
        self.assertEqual(summary["category_totals"]["Household"], 6.0)
        self.assertEqual(summary["category_totals"]["Transport"], 3.0)
        self.assertEqual(summary["top_stores"]["Market"], 1)

    def test_build_dashboard_summary_returns_metrics(self) -> None:
        summary = build_dashboard_summary(sample_history())
        self.assertEqual(summary["grand_total"], 13.0)
        self.assertEqual(summary["top_category"], "Household")
        self.assertEqual(len(summary["receipt_labels"]), 2)

    def test_budget_source_skips_unparseable_dates(self) -> None:
        budget_source = build_budget_source_data(sample_history())
        self.assertEqual(budget_source["skipped_receipts"], 1)
        self.assertEqual(len(budget_source["receipts_df"]), 1)

    def test_normalize_selected_budget_month_uses_latest_receipt_month(self) -> None:
        budget_source = build_budget_source_data(sample_history())
        normalized = normalize_selected_budget_month(budget_source["receipts_df"], "")
        self.assertEqual(normalized, "2025-03")

    def test_build_month_budget_analytics_uses_budget_and_actuals(self) -> None:
        budget_source = build_budget_source_data(sample_history())
        analytics = build_month_budget_analytics(
            budget_source,
            {"Food": 20.0, "Household": 10.0},
            "2025-03",
        )
        category_df = analytics["category_df"].set_index("Category")
        self.assertEqual(analytics["total_budget"], 30.0)
        self.assertEqual(analytics["total_actual"], 10.0)
        self.assertEqual(category_df.loc["Food", "Actual"], 4.0)
        self.assertEqual(category_df.loc["Household", "Actual"], 6.0)


if __name__ == "__main__":
    unittest.main()
