import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# ------------------------------------
# Page Config
# ------------------------------------

st.set_page_config(
    page_title="Multi PDF RAG Assistant",
    layout="wide"
)

st.title("📚 Multi PDF RAG Assistant")

# ------------------------------------
# Session State
# ------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

# ------------------------------------
# Upload PDFs
# ------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

# ------------------------------------
# Build Vector Store
# ------------------------------------

if uploaded_files:

    current_files = sorted(
        [file.name for file in uploaded_files]
    )

    if current_files != st.session_state.processed_files:

        with st.spinner("Processing PDFs..."):

            all_docs = []

            for uploaded_file in uploaded_files:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:

                    tmp_file.write(
                        uploaded_file.read()
                    )

                    pdf_path = tmp_file.name

                loader = PyPDFLoader(pdf_path)

                docs = loader.load()

                # Store source metadata
                for doc in docs:
                    doc.metadata["source_pdf"] = (
                        uploaded_file.name
                    )

                all_docs.extend(docs)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(
                all_docs
            )

            st.success(
                f"{len(uploaded_files)} PDFs processed"
            )

            st.write(
                f"Pages Loaded: {len(all_docs)}"
            )

            st.write(
                f"Chunks Created: {len(chunks)}"
            )

            embeddings = OllamaEmbeddings(
                model="nomic-embed-text"
            )

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.processed_files = (
                current_files
            )

# ------------------------------------
# Ask Questions
# ------------------------------------

if st.session_state.vectorstore:

    question = st.text_input(
        "Ask a question about your PDFs"
    )

    if question:

        with st.spinner("Searching..."):

            retriever = (
                st.session_state.vectorstore
                .as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 8}
                )
            )

            retrieved_docs = (
                retriever.invoke(question)
            )

            # Debug Sources
            source_names = []

            for doc in retrieved_docs:

                source = doc.metadata.get(
                    "source_pdf",
                    "Unknown"
                )

                source_names.append(source)

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            llm = ChatOllama(
                model="llama3.2",
                temperature=0
            )

            prompt = f"""
You are a document assistant.

Use ONLY the supplied context.

If the answer is not present,
say:

"I could not find that information in the uploaded PDFs."

Context:
{context}

Question:
{question}
"""

            response = llm.invoke(prompt)

            st.markdown("## Answer")

            st.write(
                response.content
            )

            # --------------------
            # Sources
            # --------------------

            st.markdown(
                "## Retrieved Sources"
            )

            unique_sources = sorted(
                set(source_names)
            )

            for source in unique_sources:

                st.write(
                    f"📄 {source}"
                )

            # --------------------
            # Debug Chunks
            # --------------------

            with st.expander(
                "View Retrieved Chunks"
            ):

                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    st.markdown(
                        f"### Chunk {i}"
                    )

                    st.write(
                        f"Source: {doc.metadata.get('source_pdf')}"
                    )

                    st.write(
                        doc.page_content[:1200]
                    )