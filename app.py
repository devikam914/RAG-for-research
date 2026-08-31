import streamlit as st
from src.client import get_gemini_client
from src.document_processor import file_hash
from src.generator import generate_grounded_answer
from src.vector_store import build_vector_index, search_similar_chunks

st.set_page_config(
    page_title="Research RAG",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1200px; padding-top: 2rem;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(135deg,#172554,#312e81); color: white; margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .35rem 0 0; color: #dbeafe;}
    [data-testid="stSidebar"] {background-color: #f8fafc;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>📚 Research RAG</h1>'
    '<p>Ask questions grounded in your uploaded research papers.</p></div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def cached_client():
    try:
        return get_gemini_client()
    except ValueError as err:
        st.error(str(err))
        st.stop()


if "library" not in st.session_state:
    st.session_state.library = {}
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Paper library")
    uploaded = st.file_uploader(
        "Add research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="You can add multiple papers. Click the button after selecting them.",
    )
    if st.button("Add selected papers", use_container_width=True):
        if not uploaded:
            st.warning("Select at least one PDF first.")
        else:
            added = 0
            for file in uploaded:
                data = file.getvalue()
                key = file_hash(data)
                if key not in st.session_state.library:
                    st.session_state.library[key] = {"name": file.name, "bytes": data}
                    added += 1
            st.success(f"Added {added} new paper(s).")

    st.caption(f"{len(st.session_state.library)} paper(s) in library")

    if st.button("Build / rebuild index", type="primary", use_container_width=True):
        if not st.session_state.library:
            st.warning("Add papers before building the index.")
        else:
            with st.spinner("Extracting text and creating embeddings..."):
                try:
                    client = cached_client()
                    chunks, embeddings = build_vector_index(st.session_state.library, client)
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.success(f"Indexed {len(chunks)} chunks.")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    if st.session_state.chunks:
        st.info(f"Ready: {len(st.session_state.chunks)} searchable chunks")

    use_web = st.toggle(
        "Allow Google Search when needed",
        value=False,
        help="Optional real-time web grounding. Keep off for paper-only answers.",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if not st.session_state.chunks:
    st.info("Add one or more PDFs in the sidebar, then build the index to start chatting.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved paper excerpts"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']} — page {source['page']}** "
                        f"(similarity: {source['score']:.3f})\n\n{source['text']}"
                    )

query = st.chat_input("Ask about a concept, method, result, or comparison...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            try:
                client = cached_client()
                
                # ADD THIS LINE: Prove to the type checker it's not None
                assert st.session_state.embeddings is not None, "Embeddings not initialized"
                assert st.session_state.chunks is not None, "Chunks not initialized"

                sources = search_similar_chunks(
                    query=query,
                    chunks=st.session_state.chunks,
                    embeddings=st.session_state.embeddings,
                    client=client,
                )

                answer = generate_grounded_answer(
                    query=query,
                    retrieved_chunks=sources,
                    client=client,
                    use_web=use_web,
                )

                st.markdown(answer)
                with st.expander("Retrieved paper excerpts"):
                    for source in sources:
                        st.markdown(
                            f"**{source['source']} — page {source['page']}** "
                            f"(similarity: {source['score']:.3f})\n\n{source['text']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })
            except Exception as exc:
                error = f"I couldn't answer that yet: {exc}"
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "content": error})
