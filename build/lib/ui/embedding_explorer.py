"""UMAP visualization of document chunk embeddings from ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import streamlit as st

from app.core.config import settings
from app.retrieval.chroma_store import chroma_service
from ui.api_client import APIError, IDPClient


@dataclass
class EmbeddingPoint:
    x: float
    y: float
    chunk_index: int
    page_number: int | None
    chunk_type: str
    text_preview: str


def _reduce_embeddings(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    from sklearn.decomposition import PCA
    import umap

    n = len(embeddings)
    if n == 1:
        return np.array([[0.0, 0.0]]), "single-point"
    if n == 2:
        return PCA(n_components=2, random_state=42).fit_transform(embeddings), "pca"

    n_neighbors = min(15, n - 1)
    try:
        coords = umap.UMAP(
            n_components=2,
            random_state=42,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="cosine",
        ).fit_transform(embeddings)
        return coords, "umap"
    except Exception:
        return PCA(n_components=2, random_state=42).fit_transform(embeddings), "pca"


def _compute_stats(embeddings: np.ndarray) -> dict[str, float | int]:
    norms = np.linalg.norm(embeddings, axis=1)
    stats: dict[str, float | int] = {
        "chunks": len(embeddings),
        "dimensions": embeddings.shape[1],
        "mean_norm": float(norms.mean()),
    }
    if len(embeddings) >= 2:
        normed = embeddings / norms[:, np.newaxis]
        sims = normed @ normed.T
        upper = sims[np.triu_indices(len(embeddings), k=1)]
        stats["mean_cosine_similarity"] = float(upper.mean())
        stats["min_cosine_similarity"] = float(upper.min())
        stats["max_cosine_similarity"] = float(upper.max())
    return stats


def build_embedding_viz(document_id: str) -> tuple[list[EmbeddingPoint], dict, str]:
    raw = chroma_service.get_document_embeddings(document_id)
    if not raw:
        raise ValueError("No embeddings found in Chroma for this document.")

    embeddings = np.array(raw["embeddings"], dtype=np.float32)
    coords, method = _reduce_embeddings(embeddings)
    stats = _compute_stats(embeddings)
    stats["collection"] = raw["collection_name"]

    points: list[EmbeddingPoint] = []
    for i, meta in enumerate(raw["metadatas"]):
        text = raw["documents"][i] if raw["documents"] else ""
        points.append(
            EmbeddingPoint(
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                chunk_index=int(meta.get("chunk_index", i)),
                page_number=meta.get("page_number"),
                chunk_type=str(meta.get("chunk_type", "text")),
                text_preview=text[:120].replace("\n", " ") + ("…" if len(text) > 120 else ""),
            )
        )

    return points, stats, method


def _plot_points(points: list[EmbeddingPoint], color_by: str):
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(
        {
            "x": [p.x for p in points],
            "y": [p.y for p in points],
            "page": [p.page_number if p.page_number is not None else "?" for p in points],
            "chunk_type": [p.chunk_type for p in points],
            "chunk_index": [p.chunk_index for p in points],
            "text": [p.text_preview for p in points],
        }
    )
    color_col = "page" if color_by == "Page" else "chunk_type"
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_col,
        hover_data=["chunk_index", "text"],
        title="Chunk embeddings (2D projection)",
        height=560,
    )
    fig.update_layout(
        xaxis_title="dim 1",
        yaxis_title="dim 2",
        legend_title=color_col.replace("_", " ").title(),
    )
    return fig


def render_embeddings_tab(client: IDPClient) -> None:
    st.header("Embedding explorer")
    st.caption(
        f"Visualize chunk embeddings from ChromaDB "
        f"({settings.openai_embedding_model}, cosine space)."
    )

    try:
        docs = client.list_documents()
    except APIError as e:
        st.error(e.detail)
        return

    ready_docs = [d for d in docs if d["status"] == "ready"]
    if not ready_docs:
        st.warning("No ready documents yet. Upload and process a document first.")
        return

    labels = {f"{d['filename']} [{str(d['document_id'])[:8]}…]": d for d in ready_docs}
    picked_label = st.selectbox("Document", options=list(labels.keys()), key="embed_doc_pick")
    doc = labels[picked_label]
    doc_id = doc["document_id"]

    col1, col2 = st.columns(2)
    with col1:
        color_by = st.radio("Color by", ["Page", "Chunk type"], horizontal=True, key="embed_color")
    with col2:
        run = st.button("Generate plot", type="primary", key="embed_run")

    if not run and "embed_viz_cache" not in st.session_state:
        st.info("Select a document and click **Generate plot**.")
        return

    if run or st.session_state.get("embed_viz_doc_id") != doc_id:
        with st.spinner("Loading embeddings and running UMAP…"):
            try:
                points, stats, method = build_embedding_viz(doc_id)
                st.session_state.embed_viz_cache = (points, stats, method)
                st.session_state.embed_viz_doc_id = doc_id
            except ValueError as e:
                st.error(str(e))
                return
            except ImportError as e:
                st.error(
                    "Missing visualization packages. Run:\n\n"
                    "`uv sync --extra ui`"
                )
                st.code(str(e))
                return

    points, stats, method = st.session_state.embed_viz_cache

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chunks", stats["chunks"])
    m2.metric("Dimensions", stats["dimensions"])
    m3.metric("Projection", method.upper())
    if "mean_cosine_similarity" in stats:
        m4.metric("Avg similarity", f"{stats['mean_cosine_similarity']:.3f}")
    else:
        m4.metric("Avg similarity", "—")

    st.plotly_chart(_plot_points(points, color_by), use_container_width=True)

    with st.expander("Similarity stats"):
        st.json(stats)

    with st.expander("Sample chunks"):
        for p in sorted(points, key=lambda x: x.chunk_index)[:8]:
            st.markdown(
                f"**#{p.chunk_index}** · p.{p.page_number} · `{p.chunk_type}` — {p.text_preview}"
            )
