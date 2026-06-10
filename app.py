"""Streamlit chatbot UI — a grounded, document-backed assistant over the 10-Ks.

    streamlit run app.py

Requires indexes built first (python -m scripts.build_index).
"""
from __future__ import annotations

import streamlit as st

import config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Financial Filings RAG", page_icon="📑", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline(strategy: str) -> RAGPipeline:
    return RAGPipeline(strategy=strategy)


def index_exists(strategy: str) -> bool:
    return (config.INDEX_DIR / strategy / "chunks.json").exists()


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
    strategy = st.selectbox("Chunking strategy", available, index=len(available) - 1)
    use_rerank = st.toggle("Rerank candidates", value=True)
    top_k = st.slider("Sources (top-k)", 1, 10, config.TOP_K)
    st.divider()
    st.markdown("**Corpus**")
    for t, name in config.COMPANIES.items():
        st.markdown(f"- `{t}` — {name}")
    if not config.OPENAI_API_KEY:
        st.warning("OPENAI_API_KEY not set — generation will fail. Add it to .env.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['n']}] {s['company']} ({s['ticker']})** — "
                        f"[filing]({s['source_url']})  \n"
                        f"`{s['chunk_id']}`\n\n> {s['snippet']}"
                    )

if prompt := st.chat_input("e.g. What were Apple's total net sales last year?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and reading filings…"):
            pipe = load_pipeline(strategy)
            result = pipe.answer(prompt, use_rerank=use_rerank, top_k=top_k)
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
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(
                        f"**[{s['n']}] {s['company']} ({s['ticker']})** — "
                        f"[filing]({s['source_url']})  \n"
                        f"`{s['chunk_id']}`\n\n> {s['snippet']}"
                    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.answer.text,
            "sources": [] if result.answer.refused else sources,
        }
    )
