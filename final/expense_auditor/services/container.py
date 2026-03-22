from __future__ import annotations

from dataclasses import dataclass

from expense_auditor.services.audio import AudioService
from expense_auditor.services.gemini.client import GeminiClient
from expense_auditor.services.gemini.insights import GeminiInsightService
from expense_auditor.services.gemini.receipt_analysis import ReceiptAnalysisService
from expense_auditor.services.rag import RagService


@dataclass(frozen=True)
class AppServices:
    gemini_client: GeminiClient
    receipt_analysis: ReceiptAnalysisService
    insights: GeminiInsightService
    audio: AudioService
    rag: RagService

