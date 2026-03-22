from __future__ import annotations

import streamlit as st

from expense_auditor.constants import DEFAULT_CATEGORIES
from expense_auditor.parsers.categories import parse_categories
from expense_auditor.persistence.session_repository import SessionRepository
from expense_auditor.services.container import AppServices
from expense_auditor.ui.sidebar import render_sidebar
from expense_auditor.ui.tabs.budget import render_budget_tab
from expense_auditor.ui.tabs.dashboard import render_dashboard_tab
from expense_auditor.ui.tabs.receipts import render_receipts_tab
from expense_auditor.ui.tabs.search import render_search_tab
from expense_auditor.ui.tabs.suggestions import render_suggestions_tab


def render_app(repo: SessionRepository, services: AppServices) -> None:
    api_key, categories_file = render_sidebar(repo)

    if not api_key:
        st.header("Step 1 — Set up an API key 🔑")
        st.info("Enter your Gemini API key in the sidebar.")

    categories: list[str] = []
    categories_valid = False

    if api_key and not categories_file:
        st.header("Step 2 — Upload your categories file in the sidebar.")
        st.caption("`.txt` with one category per line, or `.csv` with a `category` column.")
        st.info("If you don't have a custom file, click **Use default categories** below.")
        st.markdown("The default categories are:")
        st.markdown("- " + "\n- ".join(DEFAULT_CATEGORIES))

        if repo.get_categories_signature() != "__default__":
            repo.invalidate_categories()

        if not repo.get_categories_approved():
            if st.button("✅ Use default expense categories", type="primary", use_container_width=True):
                repo.approve_categories(DEFAULT_CATEGORIES, "__default__")
                repo.clear_generated_insights()
                st.rerun()
            st.info("Please upload a categories file or use the default categories to continue.")
        else:
            categories = DEFAULT_CATEGORIES
            categories_valid = True
            repo.set_approved_categories(categories)
            st.success("Using default categories. Continue with receipt upload below.")

    elif api_key and categories_file:
        try:
            categories = parse_categories(categories_file)
        except Exception as exc:
            st.error(f"Could not read categories file: {exc}")
            categories = []

        if categories:
            current_signature = f"{categories_file.name}:{categories_file.size}"
            if repo.get_categories_signature() != current_signature:
                repo.invalidate_categories(current_signature)

            st.markdown(f"**{len(categories)} categories loaded from file:**")
            for category in categories:
                st.markdown(f"- {category}")

            categories_valid = True
            if not repo.get_categories_approved():
                if st.button("✅ Approve Categories", type="primary", use_container_width=True):
                    repo.approve_categories(categories, current_signature)
                    repo.clear_generated_insights()
                    st.rerun()
                st.info("Please approve these categories to continue.")
            else:
                repo.set_approved_categories(categories)
                st.success("Categories approved. Continue with receipt upload below.")

    if not (api_key and categories_valid and repo.get_categories_approved()):
        st.stop()

    services.gemini_client.configure(api_key)
    categories = repo.get_approved_categories()
    repo.sync_category_budgets(categories)

    st.divider()
    tab_chat, tab_dash, tab_suggestions, tab_budget, tab_search = st.tabs(
        ["📨 Receipts", "📊 Dashboard", "💡 Suggestions", "💸 Budget", "🔍 Receipt Search"]
    )

    with tab_chat:
        render_receipts_tab(repo, services, categories)

    with tab_dash:
        render_dashboard_tab(repo)

    with tab_suggestions:
        render_suggestions_tab(repo, services, categories)

    with tab_budget:
        render_budget_tab(repo, categories)

    with tab_search:
        render_search_tab(repo, services)

