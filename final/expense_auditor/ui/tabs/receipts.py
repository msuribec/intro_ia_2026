from __future__ import annotations

import json

import streamlit as st

from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.container import AppServices
from expense_auditor.ui.components.receipt_result import render_receipt_result


def render_receipts_tab(
    repo: SessionRepository,
    services: AppServices,
    categories: list[str],
) -> None:
    history = repo.get_receipt_history()

    for index, entry in enumerate(history):
        with st.chat_message("user"):
            if entry.get("image_bytes"):
                st.image(entry["image_bytes"], width=260)
            else:
                st.info("Imported from CSV")
        with st.chat_message("assistant"):
            render_receipt_result(index, entry, categories, repo, services)

    st.divider()
    st.markdown("#### Upload a new receipt")
    receipt_file = st.file_uploader(
        "Photo of your receipt or invoice (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"receipt_file_{repo.get_widget_seed()}_{len(history)}",
    )

    if receipt_file:
        st.image(receipt_file, caption="Preview", width=260)
        if st.button("\U0001f50d Extract data from receipt", type="primary", use_container_width=True):
            with st.spinner("Extracting receipt data with Gemini Vision..."):
                try:
                    data = services.receipt_analysis.analyze_receipt(
                        receipt_file.getvalue(),
                        categories,
                    )
                except json.JSONDecodeError:
                    st.error("The model returned an unexpected response. Please try again.")
                    st.stop()
                except Exception as exc:
                    st.error(f"Error: {exc}")
                    st.stop()

            data["savings_tip"] = ""
            data["tip_language"] = ""

            repo.append_receipt(
                {
                    "image_bytes": receipt_file.getvalue(),
                    "data": data,
                    "audio_bytes": None,
                }
            )
            repo.clear_generated_insights()
            st.rerun()

