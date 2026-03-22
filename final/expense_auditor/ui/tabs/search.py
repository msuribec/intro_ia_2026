from __future__ import annotations

import streamlit as st

from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.container import AppServices


def render_search_tab(repo: SessionRepository, services: AppServices) -> None:
    history = repo.get_receipt_history()
    if not history:
        st.info("No receipts yet. Analyze some receipts in the **Receipts** tab first.")
        return

    if repo.get_vector_store() is None or repo.get_rag_receipt_count() != len(history):
        with st.spinner(f"Indexing {len(history)} receipt(s)…"):
            store, first_error = services.rag.build_vector_store(history)

        if store.ntotal == 0:
            st.error(
                f"Could not index any receipts. Error: {first_error or 'Unknown error'}. "
                "Check that your API key has access to the Gemini embedding model "
                "and that the embedding dimensionality matches the FAISS index."
            )
            repo.set_vector_store(None)
            repo.set_rag_receipt_count(0)
            st.stop()

        repo.set_vector_store(store)
        repo.set_rag_receipt_count(len(history))
        if first_error:
            st.warning(f"Some receipts could not be indexed: {first_error}")

    vector_store = repo.get_vector_store()

    st.markdown("#### Ask about your receipts")
    for msg in repo.get_rag_chat_history():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("E.g. How much did I spend on groceries?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        repo.append_rag_chat_message("user", question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = services.rag.answer_question(question, vector_store)
                except Exception as exc:
                    answer = f"Error generating answer: {exc}"
            st.markdown(answer)
        repo.append_rag_chat_message("assistant", answer)

    if repo.get_rag_chat_history():
        if st.button("Clear chat history", use_container_width=False):
            repo.clear_rag_chat_history()
            st.rerun()

