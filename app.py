"""Streamlit chatbot UI — a grounded, document-backed assistant over the 10-Ks.

    streamlit run app.py

Requires indexes built first (python -m scripts.build_index).
"""
from __future__ import annotations

import hmac

import streamlit as st

import config
from src.pipeline import RAGPipeline
from src.telemetry import log_feedback, setup_logging

setup_logging()
st.set_page_config(page_title="Financial Filings RAG", page_icon="📑", layout="wide")


def check_password() -> bool:
    if not config.APP_PASSWORD:
        return True
    if st.session_state.get("authed"):
        return True
    st.title("📑 Financial Document Intelligence")
    pw = st.text_input("Password", type="password")
    if pw:
        if hmac.compare_digest(pw, config.APP_PASSWORD):
            st.session_state["authed"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False


if not check_password():
    st.stop()


@st.cache_resource(show_spinner=False)
def load_pipeline(strategy: str) -> RAGPipeline:
    return RAGPipeline(strategy=strategy)


def index_exists(strategy: str) -> bool:
    return (config.INDEX_DIR / strategy / "chunks.json").exists()


def _on_feedback(idx: int) -> None:
    rating = st.session_state.get(f"feedback_{idx}")
    msg = st.session_state.messages[idx]
    log_feedback(
        {
            "query": msg.get("query", ""),
            "answer_snippet": msg["content"][:300],
            "rating": "up" if rating == 1 else "down",
        }
    )


st.title("📑 Financial Document Intelligence")
st.caption(
    "Ask questions across the latest 10-K filings for "
    + ", ".join(config.COMPANIES) + ". Answers are grounded and cited."
)

with st.sidebar:
    st.header("Settings")
    available = [s for s in config.STRATEGIES if index_exists(s)]
    if not available:
        st.error("No index found. Run `python -m scripts.build_index` first.")
        st.stop()
    # Chunking strategy is an internal retrieval detail, not a user-facing knob:
    # prefer semantic (best in the eval), else whatever index is built.
    strategy = "semantic" if "semantic" in available else available[-1]
    model = st.selectbox("Answer model", config.CHAT_MODELS, index=0)
    use_rerank = st.toggle("Rerank candidates", value=True)
    top_k = st.slider("Sources (top-k)", 1, 10, config.TOP_K)
    st.divider()
    st.markdown("**Corpus**")
    for t, name in config.COMPANIES.items():
        st.markdown(f"- `{t}` — {name}")
    if not config.OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set — generation will fail. Add it to .env.")
    if not config.PINECONE_API_KEY:
        st.warning("PINECONE_API_KEY not set — retrieval will fail. Add it to .env.")
    if not config.APP_PASSWORD:
        st.warning("APP_PASSWORD not set — UI is open (dev mode).")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(sources: list[dict]) -> None:
    with st.expander("Sources"):
        for s in sources:
            st.markdown(
                f"**[{s['n']}] {s['company']} ({s['ticker']})** — "
                f"[filing]({s['source_url']})  \n"
                f"`{s['chunk_id']}`\n\n> {s['snippet']}"
            )


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])
        if msg.get("stats"):
            st.caption(msg["stats"])
        if msg["role"] == "assistant":
            st.feedback("thumbs", key=f"feedback_{i}", on_change=_on_feedback, args=(i,))

if prompt := st.chat_input("e.g. What were Apple's total net sales last year?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reading filings…"):
            pipe = load_pipeline(strategy)
            result = pipe.answer(prompt, use_rerank=use_rerank, top_k=top_k, model=model)
        st.markdown(result.answer.text)
        sources = [
            {
                "n": s.n,
                "ticker": s.ticker,
                "company": s.company,
                "source_url": s.source_url,
                "chunk_id": s.chunk_id,
                "snippet": (s.text[:400] + "…") if len(s.text) > 400 else s.text,
            }
            for s in result.answer.sources
        ]
        if sources and not result.answer.refused:
            render_sources(sources)
        stats = None
        if result.trace:
            t = result.trace
            stats = (
                f"{t.total_ms/1000:.1f}s total · retrieval {t.retrieval_ms:.0f}ms · "
                f"rerank {t.rerank_ms:.0f}ms · generation {t.generation_ms/1000:.1f}s · "
                f"~${t.est_cost_usd:.4f}"
            )
            st.caption(stats)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer.text,
            "query": prompt,
            "sources": [] if result.answer.refused else sources,
            "stats": stats,
        }
    )
    st.rerun()
