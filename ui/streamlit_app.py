"""
IDP RAG — Streamlit UI

Run (API + worker must be running):
    uv run streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from app.core.config import settings
from ui.api_client import APIError, IDPClient
from ui.embedding_explorer import render_embeddings_tab

STATUS_COLORS = {
    "ready": "🟢",
    "failed": "🔴",
    "queued": "🟡",
    "parsing": "🔵",
    "chunking": "🔵",
    "embedding": "🔵",
    "enriching": "🔵",
}


def get_client() -> IDPClient:
    return IDPClient(base_url=st.session_state.api_url)


def init_state() -> None:
    defaults = {
        "api_url": settings.api_public_url,
        "session_id": None,
        "chat_history": [],
        "active_doc_id": None,
        "watch_doc_id": None,
        "upload_in_progress": False,
        "query_scope": "Single document",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _doc_label(doc: dict) -> str:
    return f"{doc['filename']} [{str(doc['document_id'])[:8]}…]"


def _doc_maps(ready_docs: list[dict]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """label_to_id, id_to_label, all_labels."""
    label_to_id = {_doc_label(d): d["document_id"] for d in ready_docs}
    id_to_label = {d["document_id"]: _doc_label(d) for d in ready_docs}
    return label_to_id, id_to_label, list(label_to_id.keys())


def _set_active_document(doc_id: str, id_to_label: dict[str, str]) -> None:
    """Switch chat focus to one document (Documents tab → Chat)."""
    st.session_state.active_doc_id = doc_id
    label = id_to_label.get(doc_id)
    if label:
        st.session_state.chat_single_doc = label
        st.session_state.chat_multi_docs = [label]


def _resolve_query_doc_ids(
    ready_docs: list[dict],
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Render document picker widgets and return selected IDs for the query API."""
    label_to_id, id_to_label, all_labels = _doc_maps(ready_docs)
    if not all_labels:
        return [], label_to_id, id_to_label

    active = st.session_state.get("active_doc_id")
    if active and active in id_to_label and "chat_single_doc" not in st.session_state:
        st.session_state.chat_single_doc = id_to_label[active]
    if active and active in id_to_label and "chat_multi_docs" not in st.session_state:
        st.session_state.chat_multi_docs = [id_to_label[active]]
    if "chat_single_doc" not in st.session_state:
        st.session_state.chat_single_doc = all_labels[0]
    if "chat_multi_docs" not in st.session_state:
        st.session_state.chat_multi_docs = [all_labels[0]]

    scope = st.radio(
        "Query scope",
        ["Single document", "Multiple documents"],
        horizontal=True,
        key="query_scope",
    )

    if scope == "Single document":
        picked = st.selectbox(
            "Document to question",
            options=all_labels,
            key="chat_single_doc",
            help="Pick any ready document — no need to delete others",
        )
        selected_ids = [label_to_id[picked]]
        st.session_state.active_doc_id = selected_ids[0]
    else:
        picked_labels = st.multiselect(
            "Documents to question",
            options=all_labels,
            key="chat_multi_docs",
            help="Select one or more documents for cross-document questions",
        )
        selected_ids = [label_to_id[l] for l in picked_labels]

    return selected_ids, label_to_id, id_to_label


def render_sidebar() -> None:
    with st.sidebar:
        st.title("IDP RAG")
        st.caption("Document intelligence + RAG Q&A")

        st.session_state.api_url = st.text_input(
            "API URL",
            value=st.session_state.api_url,
            help="FastAPI server address",
        )

        try:
            client = get_client()
            health = client.health()
            st.success(f"API: {health.get('status', 'ok')}")
        except Exception as e:
            st.error(f"API unreachable: {e}")
            st.info(f"Start API:\n`./scripts/start_api.sh` (port {settings.api_port})")

        st.divider()
        st.markdown("**Services needed**")
        st.code(
            "# Terminal 1 — API\n"
            f"./scripts/start_api.sh  # port {settings.api_port}\n\n"
            "# Terminal 2 — Worker\n"
            "./scripts/start_worker.sh",
            language="bash",
        )


def _render_processing_banner(client: IDPClient, doc_id: str) -> None:
    """Prominent live status for the document just uploaded."""
    try:
        status = client.document_status(doc_id)
    except APIError as e:
        st.warning(f"Could not load status: {e.detail}")
        return

    stage = status.get("status", "queued")
    progress = status.get("progress") or 0.0
    icon = STATUS_COLORS.get(stage, "⚪")

    st.info(f"{icon} **Processing** — stage: `{stage}`")
    st.progress(progress, text=f"{int(progress * 100)}%")
    if stage == "queued":
        st.caption(
            "Waiting for the worker to start — if this lasts more than a minute, "
            "restart the Celery worker (`uv run python -m worker`)."
        )
    elif stage == "embedding":
        st.caption("Embedding chunks — large documents can take several minutes.")
    if status.get("error_message"):
        st.error(status["error_message"])


def _render_document_list(docs: list[dict], client: IDPClient) -> None:
    if not docs:
        st.info("No documents yet. Upload a PDF or DOCX above.")
        return

    for doc in docs:
        icon = STATUS_COLORS.get(doc["status"], "⚪")
        entities_status = doc.get("entities_status")
        entities_note = ""
        if doc["status"] == "ready" and entities_status == "pending":
            entities_note = " · entities extracting…"
        elif doc["status"] == "ready" and entities_status == "complete":
            entities_note = " · entities ready"
        elif doc["status"] == "ready" and entities_status == "failed":
            entities_note = " · entity extraction failed"
        with st.expander(
            f"{icon} {doc['filename']}  —  {doc['status']}{entities_note}",
            expanded=doc["status"] not in ("ready", "failed"),
        ):
            st.write(f"**ID:** `{doc['document_id']}`")
            st.write(f"**Type:** {doc.get('doc_type') or '—'}")
            st.write(f"**Pages:** {doc.get('page_count') or '—'}")

            if doc["status"] not in ("ready", "failed"):
                try:
                    status = client.document_status(doc["document_id"])
                    progress = status.get("progress") or 0.0
                    st.progress(progress)
                    stage = doc["status"]
                    if stage == "embedding":
                        st.caption(
                            f"Embedding chunks ({int(progress * 100)}%) — "
                            "large documents can take several minutes here."
                        )
                    if status.get("error_message"):
                        st.error(status["error_message"])
                except APIError as e:
                    st.warning(e.detail)

            cols = st.columns(2)
            with cols[0]:
                if doc["status"] == "ready":
                    if st.button("Select for chat", key=f"sel_{doc['document_id']}"):
                        doc_id = doc["document_id"]
                        label = _doc_label(doc)
                        st.session_state.active_doc_id = doc_id
                        st.session_state.chat_single_doc = label
                        st.session_state.chat_multi_docs = [label]
                        st.session_state.query_scope = "Single document"
                        st.toast(f"Selected **{doc['filename']}** — open the Chat tab")
            with cols[1]:
                if st.button("Delete", key=f"del_{doc['document_id']}"):
                    try:
                        client.delete_document(doc["document_id"])
                        doc_id = doc["document_id"]
                        if st.session_state.get("active_doc_id") == doc_id:
                            st.session_state.active_doc_id = None
                        st.rerun()
                    except APIError as e:
                        st.error(e.detail)


@st.fragment(run_every=2)
def _live_documents_panel() -> None:
    """Refresh document list + progress every 2s while on this tab."""
    client = get_client()
    watch_id = st.session_state.get("watch_doc_id")

    if watch_id:
        _render_processing_banner(client, watch_id)

    try:
        docs = client.list_documents()
    except APIError as e:
        st.error(e.detail)
        return

    _render_document_list(docs, client)

    if watch_id:
        try:
            status = client.document_status(watch_id)
            if status.get("status") == "ready":
                label = None
                for d in docs:
                    if d["document_id"] == watch_id:
                        label = _doc_label(d)
                        break
                st.session_state.active_doc_id = watch_id
                if label:
                    st.session_state.chat_single_doc = label
                    st.session_state.chat_multi_docs = [label]
                st.session_state.pop("watch_doc_id", None)
            elif status.get("status") == "failed":
                st.session_state.pop("watch_doc_id", None)
        except APIError:
            pass


def render_documents_tab() -> None:
    st.header("Documents")
    client = get_client()

    with st.form("upload_form", clear_on_submit=False):
        uploaded = st.file_uploader(
            "Upload document",
            type=["pdf", "docx"],
            key="doc_upload",
        )
        submitted = st.form_submit_button(
            "Upload & process",
            type="primary",
            disabled=st.session_state.get("upload_in_progress", False),
        )

    if submitted:
        if not uploaded:
            st.error("Choose a PDF or DOCX file first.")
        else:
            st.session_state.upload_in_progress = True
            try:
                with st.spinner("Uploading…"):
                    result = client.upload_document(uploaded.name, uploaded.getvalue())
                doc_id = str(result["document_id"])
                st.session_state.watch_doc_id = doc_id
                st.session_state.active_doc_id = doc_id
                st.success(f"Queued: **{result['filename']}**")
            except APIError as e:
                st.error(e.detail)
            finally:
                st.session_state.upload_in_progress = False
            st.rerun()

    st.divider()

    if st.button("Refresh list", key="refresh_docs"):
        st.rerun()

    auto = st.checkbox("Auto-refresh while processing", value=True, key="auto_refresh")
    if auto:
        _live_documents_panel()
    else:
        watch_id = st.session_state.get("watch_doc_id")
        if watch_id:
            _render_processing_banner(client, watch_id)
        try:
            docs = client.list_documents()
        except APIError as e:
            st.error(e.detail)
            return
        _render_document_list(docs, client)


def _turns_from_ui_history(history: list[dict]) -> list[dict]:
    """Convert UI chat messages to API chat_history turns (prior turns only)."""
    turns = []
    i = 0
    while i < len(history):
        if history[i]["role"] == "user":
            q = history[i]["content"]
            a = ""
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                a = history[i + 1]["content"]
                i += 2
            else:
                i += 1
            turns.append({"question": q, "answer": a})
        else:
            i += 1
    return turns


def _is_failed_answer(answer: str) -> bool:
    if not answer or not answer.strip():
        return True
    markers = (
        "couldn't generate an answer",
        "don't have enough relevant information",
        "no answer generated",
    )
    lowered = answer.lower()
    return any(m in lowered for m in markers)


def render_chat_tab() -> None:
    st.header("Ask your documents")

    client = get_client()
    try:
        docs = client.list_documents()
    except APIError as e:
        st.error(e.detail)
        return

    ready_docs = [d for d in docs if d["status"] == "ready"]
    if not ready_docs:
        st.warning("No ready documents yet. Upload a document and wait for status **ready**.")
        return

    with st.expander("Document selection & settings", expanded=not st.session_state.chat_history):
        selected_ids, label_to_id, id_to_label = _resolve_query_doc_ids(ready_docs)
        use_stream = st.toggle("Stream response", value=True, key="use_stream")
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.session_id = None
            st.rerun()

    if not selected_ids:
        st.info(
            "Select at least one ready document above to start chatting. "
            "Use **Single document** mode to pick any one file without deleting others."
        )
        return

    if len(selected_ids) == 1:
        st.caption(f"Querying: **{id_to_label[selected_ids[0]]}**")
    else:
        names = ", ".join(id_to_label[i] for i in selected_ids)
        st.caption(f"Querying {len(selected_ids)} documents: {names}")

    # Render conversation history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("Sources"):
                    for c in msg["citations"]:
                        page = c.get("page", "?")
                        st.caption(f"Page {page} · score {c.get('score', 0):.2f}")
                        st.text(c.get("snippet", "")[:300])

    if not st.session_state.chat_history:
        st.caption("Ask anything about your documents. Follow-up questions work too.")

    # ChatGPT-style input — clears after each send
    question = st.chat_input("Ask a question or follow-up…")
    if not question:
        return

    st.session_state.chat_history.append({"role": "user", "content": question})

    # Prior turns only (exclude the question we just appended)
    prior_turns = _turns_from_ui_history(st.session_state.chat_history[:-1])

    with st.chat_message("assistant"):
        try:
            stream_error = None
            debug_info = {}
            if use_stream:
                placeholder = st.empty()
                status_ph = st.empty()
                full_answer = ""
                citations = []

                for event in client.query_stream(
                    question,
                    selected_ids,
                    session_id=st.session_state.session_id,
                    chat_history=prior_turns,
                ):
                    if event.get("type") == "status":
                        status_ph.info(f"Step: **{event['status']}**")
                    elif event.get("type") == "debug":
                        debug_info = event
                        status_ph.caption(
                            f"Retrieved: {event.get('vector', 0)} vector · "
                            f"{event.get('bm25', 0)} bm25 · "
                            f"{event.get('reranked', 0)} reranked"
                        )
                    elif event.get("type") == "error":
                        stream_error = event.get("message", "Unknown error")
                        break
                    elif event.get("type") == "token":
                        full_answer += event["content"]
                        placeholder.markdown(full_answer + "▌")
                    elif event.get("type") == "citations":
                        citations = event.get("citations", [])
                        if event.get("session_id"):
                            st.session_state.session_id = event["session_id"]
                        status_ph.empty()
                    elif event.get("type") == "done":
                        break

                if stream_error:
                    st.error(f"Query failed: {stream_error}")
                    st.warning("**Retry:** turn off streaming, re-select the document, or re-upload the file.")
                    st.session_state.chat_history.pop()
                    return

                placeholder.markdown(full_answer or "_No answer generated._")
                answer = full_answer
            else:
                with st.spinner("Thinking…"):
                    result = client.query(
                        question,
                        selected_ids,
                        session_id=st.session_state.session_id,
                        chat_history=prior_turns,
                    )
                st.session_state.session_id = result.get("session_id")
                answer = result["answer"]
                citations = result.get("citations", [])
                debug_info = {
                    "reranked": len(citations),
                    "confidence": result.get("confidence"),
                }
                st.markdown(answer)
                st.caption(
                    f"{result.get('latency_ms', 0) / 1000:.1f}s · "
                    f"confidence {result.get('confidence', 0):.2f} · "
                    f"sources {len(citations)}"
                )

            if _is_failed_answer(answer):
                st.warning(
                    "**No useful answer was returned.** Try turning off streaming, "
                    "re-selecting the document, or asking a more specific question."
                )
                if debug_info:
                    st.caption(
                        f"Retrieval debug — reranked chunks: {debug_info.get('reranked', len(citations))}"
                    )
            elif not citations:
                st.info("Answer generated but no source citations were attached.")

            if citations:
                with st.expander("Sources"):
                    for c in citations:
                        page = c.get("page", "?")
                        st.caption(f"Page {page} · score {c.get('score', 0):.2f}")
                        st.text(c.get("snippet", "")[:300])

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer, "citations": citations}
            )

        except APIError as e:
            st.error(e.detail)
            st.warning("**Retry:** check the document has indexed chunks (Embeddings tab), then ask again.")
            st.session_state.chat_history.pop()
            return

    st.rerun()


def render_entities_tab() -> None:
    st.header("Extracted entities")

    client = get_client()
    try:
        docs = client.list_documents()
    except APIError as e:
        st.error(e.detail)
        return

    ready_docs = [d for d in docs if d["status"] == "ready"]
    if not ready_docs:
        st.warning("No ready documents.")
        return

    options = {_doc_label(d): d["document_id"] for d in ready_docs}
    choice = st.selectbox("Document", options=list(options.keys()), key="entity_doc")
    doc_id = options[choice]
    selected = next(d for d in ready_docs if d["document_id"] == doc_id)
    entities_status = selected.get("entities_status")

    if entities_status == "pending":
        st.info("Entity extraction is still running in the background. Check back shortly.")
        if st.button("Refresh entities", key="refresh_entities_pending"):
            st.rerun()
        return

    if entities_status == "failed":
        st.warning("Entity extraction failed for this document. Chat still works.")

    if st.button("Load entities", key="load_entities"):
        try:
            data = client.get_entities(doc_id)
            status = data.get("entities_status")
            if status == "pending":
                st.info("Entity extraction is still running. Try again in a moment.")
                return
            if status == "failed":
                st.warning("Entity extraction failed for this document.")

            st.write(f"**Total entities:** {data.get('total_entities', 0)}")

            entities = data.get("entities", {})
            if not entities:
                if status == "skipped":
                    st.info("Entity extraction is disabled (NER_ENABLED=false).")
                else:
                    st.info("No entities found.")
                return

            for entity_type, items in entities.items():
                with st.expander(f"{entity_type} ({len(items)})", expanded=True):
                    for item in items:
                        st.write(
                            f"- **{item.get('text')}** "
                            f"(p.{item.get('page', '?')}, "
                            f"conf {item.get('confidence', 0):.2f})"
                        )
        except APIError as e:
            st.error(e.detail)


def main() -> None:
    st.set_page_config(
        page_title="IDP RAG",
        page_icon="📄",
        layout="wide",
    )
    init_state()
    render_sidebar()

    tab_docs, tab_chat, tab_entities, tab_embeddings = st.tabs(
        ["Documents", "Chat", "Entities", "Embeddings"]
    )
    with tab_docs:
        render_documents_tab()
    with tab_chat:
        render_chat_tab()
    with tab_entities:
        render_entities_tab()
    with tab_embeddings:
        render_embeddings_tab(get_client())


if __name__ == "__main__":
    main()
