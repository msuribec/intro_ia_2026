from __future__ import annotations

import numpy as np
import faiss
import google.generativeai as genai

from expense_auditor.services.gemini.client import GeminiClient
from expense_auditor.types import ReceiptEntry


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
NUMERIC_KEYWORDS = {"total", "sum", "how much", "spent", "cost", "amount", "price"}


def receipt_to_text(receipt: ReceiptEntry | dict) -> str:
    data = receipt.get("data", {})
    vendor = data.get("vendor") or "Unknown vendor"
    date = data.get("date") or "Unknown date"
    currency = data.get("currency") or ""
    total = data.get("total") or 0
    items = data.get("items") or []
    category_totals = data.get("category_totals") or {}
    savings_tip = data.get("savings_tip") or ""

    lines = [f"Receipt from {vendor} on {date}."]
    if items:
        lines.append("Items:")
        for item in items:
            name = item.get("name") or "Unknown item"
            price = item.get("price") or 0
            category = item.get("category") or "Uncategorized"
            lines.append(f"- {name} ({category}) {price} {currency}")

    lines.append(f"Total: {total} {currency}")

    if category_totals:
        cats = ", ".join(f"{key}: {value} {currency}" for key, value in category_totals.items())
        lines.append(f"Spending by category: {cats}")

    if savings_tip:
        lines.append(f"Savings tip: {savings_tip}")

    return "\n".join(lines)


def embed_text(text: str, task_type: str = "retrieval_document") -> list[float]:
    result = genai.embed_content(
        model=f"models/{EMBEDDING_MODEL}",
        content=text,
        task_type=task_type,
        output_dimensionality=EMBEDDING_DIM,
    )
    return result["embedding"]


class ReceiptVectorStore:
    EMBEDDING_DIM = EMBEDDING_DIM

    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        self._receipts: list[ReceiptEntry | dict] = []
        self._texts: list[str] = []

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def add_receipt(self, receipt: ReceiptEntry | dict) -> None:
        text = receipt_to_text(receipt)
        vec = np.array([embed_text(text, task_type="retrieval_document")], dtype="float32")
        if vec.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.EMBEDDING_DIM}, got {vec.shape[1]}"
            )
        faiss.normalize_L2(vec)
        self._index.add(vec)
        self._receipts.append(receipt)
        self._texts.append(text)

    def search(self, query: str, k: int = 5) -> list[str]:
        if self._index.ntotal == 0:
            return []
        vec = np.array([embed_text(query, task_type="retrieval_query")], dtype="float32")
        if vec.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.EMBEDDING_DIM}, got {vec.shape[1]}"
            )
        faiss.normalize_L2(vec)
        actual_k = min(k, self._index.ntotal)
        _, indices = self._index.search(vec, actual_k)
        return [self._texts[index] for index in indices[0] if index >= 0]

    def search_receipts(self, query: str, k: int = 5) -> list[ReceiptEntry | dict]:
        if self._index.ntotal == 0:
            return []
        vec = np.array([embed_text(query, task_type="retrieval_query")], dtype="float32")
        if vec.shape[1] != self.EMBEDDING_DIM:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.EMBEDDING_DIM}, got {vec.shape[1]}"
            )
        faiss.normalize_L2(vec)
        actual_k = min(k, self._index.ntotal)
        _, indices = self._index.search(vec, actual_k)
        return [self._receipts[index] for index in indices[0] if index >= 0]


def answer_question(question: str, vector_store: ReceiptVectorStore, model) -> str:
    context_texts = vector_store.search(question, k=5)

    if not context_texts:
        return "No receipts found in your history yet. Please add some receipts first."

    hybrid_note = ""
    if any(keyword in question.lower() for keyword in NUMERIC_KEYWORDS):
        relevant_receipts = vector_store.search_receipts(question, k=5)
        computed_total = sum(
            float(receipt.get("data", {}).get("total") or 0)
            for receipt in relevant_receipts
        )
        currency = (
            relevant_receipts[0].get("data", {}).get("currency", "")
            if relevant_receipts
            else ""
        )
        hybrid_note = (
            "\n[COMPUTED] Python sum of totals from the most relevant receipts: "
            f"{computed_total:.2f} {currency}\n"
        )

    context_block = "\n\n---\n\n".join(context_texts)
    prompt = (
        "You are a personal finance assistant. "
        "Answer the user's question using ONLY the receipt data provided below.\n"
        "If the answer requires calculation, compute it. "
        "If the answer cannot be determined from the data, say so clearly.\n\n"
        f"RECEIPTS:\n{context_block}\n"
        f"{hybrid_note}\n"
        f"QUESTION: {question}"
    )
    response = model.generate_content(prompt)
    return response.text.strip()


class RagService:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self._gemini_client = gemini_client

    def build_vector_store(
        self,
        history: list[ReceiptEntry],
    ) -> tuple[ReceiptVectorStore, str | None]:
        store = ReceiptVectorStore()
        first_error: str | None = None
        for receipt in history:
            try:
                store.add_receipt(receipt)
            except Exception as exc:
                if first_error is None:
                    first_error = str(exc)
        return store, first_error

    def answer_question(self, question: str, vector_store: ReceiptVectorStore) -> str:
        model = self._gemini_client.get_model()
        return answer_question(question, vector_store, model)
