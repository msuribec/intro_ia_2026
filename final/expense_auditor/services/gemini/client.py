from __future__ import annotations

from typing import Optional

import google.generativeai as genai


class GeminiClient:
    """Small wrapper to centralize Gemini configuration and model selection."""

    def __init__(self) -> None:
        self._configured_api_key: str | None = None

    def configure(self, api_key: str) -> None:
        if api_key and api_key != self._configured_api_key:
            genai.configure(api_key=api_key)
            self._configured_api_key = api_key

    def pick_supported_model(self) -> Optional[str]:
        preferred = [
            "models/gemini-2.5-flash",
            "models/gemini-2.5-flash-lite",
            "models/gemini-2.0-flash",
            "models/gemini-1.5-flash",
        ]
        try:
            models = list(genai.list_models())
        except Exception:
            return None

        supported = {
            model.name
            for model in models
            if hasattr(model, "supported_generation_methods")
            and "generateContent" in model.supported_generation_methods
        }
        for candidate in preferred:
            if candidate in supported:
                return candidate
        for name in supported:
            if "gemini" in name and "vision" not in name:
                return name
        return None

    def get_model(self):
        chosen_model = self.pick_supported_model() or "models/gemini-2.0-flash"
        return genai.GenerativeModel(chosen_model)

