import streamlit as st
import tempfile
from datetime import datetime
from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

# ------------------------------------
# Page Config
# ------------------------------------

st.set_page_config(
    page_title="Multi-Agent PDF Assistant",
    layout="wide"
)

st.title("🤖 Multi-Agent PDF Assistant")

# ------------------------------------
# Session State
# ------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []

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

            embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://host.docker.internal:11434"
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

            st.success(
                f"{len(uploaded_files)} PDFs processed"
            )

# ------------------------------------
# Agent State
# ------------------------------------

class AgentState(TypedDict):
    question: str
    route: str
    context: str
    answer: str


# ------------------------------------
# Tools
# ------------------------------------

def calculator_tool(question):

    try:
        return str(eval(question))
    except:
        return "Invalid calculation"


def date_tool():

    return datetime.now().strftime(
        "%d-%m-%Y"
    )


# ------------------------------------
# Supervisor
# ------------------------------------

def supervisor_node(state):

    question = state["question"].lower()

    pdf_keywords = [
        "attention",
        "transformer",
        "encoder",
        "decoder",
        "multi head",
        "langgraph",
        "stategraph"
    ]

    if any(
        op in question
        for op in ["+", "-", "*", "/"]
    ):
        return {"route": "tool"}

    if "date" in question:
        return {"route": "tool"}

    if any(
        word in question
        for word in pdf_keywords
    ):
        return {"route": "rag"}

    return {"route": "chat"}


def route_decision(state):

    return state["route"]
# ------------------------------------
# Chat History Display
# ------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )

# ------------------------------------
# Question Answering
# ------------------------------------

if st.session_state.vectorstore:

    question = st.chat_input(
        "Ask Anything"
    )

    if question:

        # Save User Message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}
            )
        )

        llm = ChatOllama(
    model="llama3.2",
    base_url="http://host.docker.internal:11434"
,
            temperature=0
        )

        # ------------------------------------
        # RAG Agent
        # ------------------------------------

        def rag_agent(state):

            docs = retriever.invoke(
                state["question"]
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            response = llm.invoke(
                f"""
Answer ONLY using the provided context.

If the answer is not present,
say:

I could not find that information in the uploaded PDFs.

Context:
{context}

Question:
{state["question"]}
"""
            )

            return {
                "context": context,
                "answer": response.content
            }

        # ------------------------------------
        # Chat Agent with Memory
        # ------------------------------------

        def chat_agent(state):

            history = ""

            for msg in st.session_state.messages[-10:]:

                history += (
                    f"{msg['role']}: "
                    f"{msg['content']}\n"
                )

            response = llm.invoke(
                f"""
Conversation History:

{history}

Current Question:

{state["question"]}
"""
            )

            return {
                "answer": response.content
            }

        # ------------------------------------
        # Tool Agent
        # ------------------------------------

        def tool_agent(state):

            q = state["question"]

            if any(
                op in q
                for op in ["+", "-", "*", "/"]
            ):

                return {
                    "answer": calculator_tool(q)
                }

            if "date" in q.lower():

                return {
                    "answer": date_tool()
                }

            return {
                "answer": "Tool not found"
            }

        # ------------------------------------
        # LangGraph
        # ------------------------------------

        graph = StateGraph(
            AgentState
        )

        graph.add_node(
            "supervisor",
            supervisor_node
        )

        graph.add_node(
            "rag_agent",
            rag_agent
        )

        graph.add_node(
            "chat_agent",
            chat_agent
        )

        graph.add_node(
            "tool_agent",
            tool_agent
        )

        graph.set_entry_point(
            "supervisor"
        )

        graph.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "rag": "rag_agent",
                "chat": "chat_agent",
                "tool": "tool_agent"
            }
        )

        graph.add_edge(
            "rag_agent",
            END
        )

        graph.add_edge(
            "chat_agent",
            END
        )

        graph.add_edge(
            "tool_agent",
            END
        )

        app = graph.compile()

        # ------------------------------------
        # Run Graph
        # ------------------------------------

        with st.spinner("Thinking..."):

            result = app.invoke(
                {
                    "question": question
                }
            )

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                result["answer"]
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"]
            }
        )

        st.success(
            f"Route Used: {result['route'].upper()}"
        )