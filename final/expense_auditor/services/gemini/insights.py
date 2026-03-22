from __future__ import annotations

import json

from expense_auditor.analytics.history import build_history_summary
from expense_auditor.parsers.gemini_json import parse_model_json_response
from expense_auditor.services.gemini.client import GeminiClient
from expense_auditor.types import CategorySuggestion, ReceiptData, ReceiptEntry


class GeminiInsightService:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self._gemini_client = gemini_client

    def generate_savings_tip(self, receipt_data: ReceiptData | dict) -> dict:
        model = self._gemini_client.get_model()
        receipt_payload = json.dumps(
            {
                "vendor": receipt_data.get("vendor", "Unknown"),
                "date": receipt_data.get("date", "Unknown"),
                "currency": receipt_data.get("currency", "$"),
                "items": receipt_data.get("items", []),
                "total": receipt_data.get("total", 0),
                "category_totals": receipt_data.get("category_totals", {}),
            },
            ensure_ascii=False,
        )
        prompt = f"""You are a personal finance auditor. Use the receipt data below to write a concise savings tip.

RECEIPT DATA:
{receipt_payload}

TASKS:
1. Review the spending pattern in this receipt.
2. Write ONE short, specific, actionable savings tip based on where most money was spent.
3. Write the tip in the same language as the receipt data when it is reasonably clear. If the language is unclear, write it in Spanish.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "savings_tip": "string",
  "tip_language": "ISO 639-1 code, e.g. es or en"
}}"""
        response = model.generate_content(prompt)
        return parse_model_json_response(response.text)

    def generate_category_suggestions(
        self,
        history: list[ReceiptEntry],
        categories: list[str],
    ) -> list[CategorySuggestion]:
        model = self._gemini_client.get_model()
        history_summary = json.dumps(build_history_summary(history), ensure_ascii=False)
        categories_payload = json.dumps(categories, ensure_ascii=False)
        prompt = f"""You are a personal finance auditor. Review the purchase history and the user's current categories.

CURRENT CATEGORIES:
{categories_payload}

PURCHASE HISTORY SUMMARY:
{history_summary}

TASKS:
1. Suggest up to 5 NEW categories that do not already exist in the current categories list.
2. Base the suggestions on recurring items, stores, and spending patterns from the history.
3. Keep each category concise and practical for a budgeting app.
4. For each suggestion, include a short reason mentioning the relevant items or stores.
5. If the current categories are already sufficient, return an empty list.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "suggested_categories": [
    {{"category": "string", "reason": "string"}}
  ]
}}"""
        response = model.generate_content(prompt)
        data = parse_model_json_response(response.text)
        return data.get("suggested_categories", [])

    def generate_history_tips(self, history: list[ReceiptEntry]) -> dict:
        model = self._gemini_client.get_model()
        history_summary = json.dumps(build_history_summary(history), ensure_ascii=False)
        prompt = f"""You are a personal finance auditor. Review the user's full purchase history and write practical savings tips.

PURCHASE HISTORY SUMMARY:
{history_summary}

TASKS:
1. Review the overall spending patterns across all receipts.
2. Write 3 short, specific, actionable savings tips.
3. Base the tips on repeated stores, recurring item types, or dominant spending categories.
4. Keep each tip to one sentence.
5. Write the tips in the same language as the purchase history when it is reasonably clear. If unclear, write them in Spanish.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "tips": ["string", "string", "string"],
  "tip_language": "ISO 639-1 code, e.g. es or en"
}}"""
        response = model.generate_content(prompt)
        data = parse_model_json_response(response.text)
        return {
            "tips": [
                tip.strip()
                for tip in data.get("tips", [])
                if isinstance(tip, str) and tip.strip()
            ],
            "tip_language": str(data.get("tip_language", "es")).strip() or "es",
        }

