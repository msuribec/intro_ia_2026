from __future__ import annotations

import json

import streamlit as st

from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.container import AppServices


def render_suggestions_tab(
    repo: SessionRepository,
    services: AppServices,
    categories: list[str],
) -> None:
    history = repo.get_receipt_history()
    if not history:
        st.info("No receipts analyzed yet. Analyze some receipts in the **Receipts** tab first.")
        return

    st.subheader("Suggestions from your full history")
    st.caption("Use your complete receipt history to discover better categories and broader saving opportunities.")

    suggest_col, tip_col = st.columns(2)

    if suggest_col.button(
        "Suggest new categories",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Reviewing your history to suggest new categories..."):
            try:
                suggestions = services.insights.generate_category_suggestions(history, categories)
                repo.set_history_category_suggestions(suggestions)
            except json.JSONDecodeError:
                st.error("The model returned an unexpected response for category suggestions. Please try again.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    if tip_col.button(
        "Generate history tips",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Reviewing your purchase history for savings tips..."):
            try:
                tip_payload = services.insights.generate_history_tips(history)
                tip_language = tip_payload.get("tip_language", "es")
                history_tip_entries = []
                for tip_text in tip_payload.get("tips", []):
                    audio_bytes, detected_lang = services.audio.generate_audio(tip_text, tip_language)
                    history_tip_entries.append(
                        {
                            "tip": tip_text,
                            "audio_bytes": audio_bytes,
                            "tip_language": detected_lang,
                        }
                    )
                repo.set_history_purchase_tips(history_tip_entries)
            except json.JSONDecodeError:
                st.error("The model returned an unexpected response for history tips. Please try again.")
            except Exception as exc:
                st.error(f"Error: {exc}")

    st.divider()

    st.markdown("**Suggested New Categories**")
    category_suggestions = repo.get_history_category_suggestions()
    if category_suggestions:
        for suggestion in category_suggestions:
            category_name = str(suggestion.get("category", "")).strip()
            reason = str(suggestion.get("reason", "")).strip()
            if category_name:
                st.markdown(f"**{category_name}**")
                if reason:
                    st.caption(reason)
    else:
        st.caption("No category suggestions generated yet.")

    st.divider()

    st.markdown("**Tips Based on All Purchase History**")
    history_tips = repo.get_history_purchase_tips()
    if history_tips:
        for tip_entry in history_tips:
            tip_text = tip_entry.get("tip", "")
            st.info(f"\U0001f3af {tip_text}")
            if tip_entry.get("audio_bytes"):
                st.audio(tip_entry["audio_bytes"], format="audio/mp3")
    else:
        st.caption("No history tips generated yet.")

