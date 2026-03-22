from __future__ import annotations

import io

from PIL import Image

from expense_auditor.parsers.gemini_json import parse_model_json_response
from expense_auditor.services.gemini.client import GeminiClient


class ReceiptAnalysisService:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self._gemini_client = gemini_client

    def analyze_receipt(self, image_bytes: bytes, categories: list[str]) -> dict:
        model = self._gemini_client.get_model()
        pil_image = Image.open(io.BytesIO(image_bytes))
        categories_block = "\n".join(f"  - {category}" for category in categories)
        prompt = f"""You are a personal finance auditor. Analyze the receipt in this image.

USER'S EXPENSE CATEGORIES:
{categories_block}

TASKS:
1. Extract every line item (product/service name and price).
2. Identify the vendor/store name.
3. Identify the purchase date (if visible).
4. Detect the currency symbol used in the receipt.
5. Assign each item to the closest matching category from the list above.
   Use "Other" only when no category is even remotely appropriate.
6. Sum the amounts per category.

Return ONLY a valid JSON object — no markdown fences, no extra text.
Schema:
{{
  "vendor":          "string",
  "date":            "string or Unknown",
  "currency":        "string (e.g. $, €, COP)",
  "items": [
    {{"name": "string", "price": 0.00, "category": "string"}}
  ],
  "total":            0.00,
  "category_totals": {{"Category Name": 0.00}}
}}"""
        response = model.generate_content([prompt, pil_image])
        return parse_model_json_response(response.text)

