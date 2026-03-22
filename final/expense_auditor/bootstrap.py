from __future__ import annotations

from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.audio import AudioService
from expense_auditor.services.container import AppServices
from expense_auditor.services.gemini.client import GeminiClient
from expense_auditor.services.gemini.insights import GeminiInsightService
from expense_auditor.services.gemini.receipt_analysis import ReceiptAnalysisService
from expense_auditor.services.rag import RagService
from expense_auditor.ui.app import render_app
from expense_auditor.ui.layout import configure_page


def build_services() -> AppServices:
    gemini_client = GeminiClient()
    return AppServices(
        gemini_client=gemini_client,
        receipt_analysis=ReceiptAnalysisService(gemini_client),
        insights=GeminiInsightService(gemini_client),
        audio=AudioService(),
        rag=RagService(gemini_client),
    )


def run_app() -> None:
    configure_page()
    repo = SessionRepository()
    repo.bootstrap()
    render_app(repo, build_services())
