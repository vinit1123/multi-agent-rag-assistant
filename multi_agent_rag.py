import streamlit as st
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

st.set_page_config(page_title="Multi Agent RAG")

st.title("🤖 Multi-Agent RAG Assistant")

# -----------------------------
# Build Vector Store
# -----------------------------

@st.cache_resource
def build_rag():

    loader = PyPDFLoader("data/sample.pdf")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore


vectorstore = build_rag()

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="llama3.2"
)

# -----------------------------
# State
# -----------------------------

class AgentState(TypedDict):
    question: str
    route: str
    context: str
    answer: str


# -----------------------------
# Supervisor
# -----------------------------

def supervisor_node(state):

    question = state["question"].lower()

    pdf_keywords = [
        "attention",
        "transformer",
        "encoder",
        "decoder",
        "multi head"
    ]

    if any(
        word in question
        for word in pdf_keywords
    ):
        return {"route": "rag"}

    return {"route": "chat"}


# -----------------------------
# Router
# -----------------------------

def route_decision(state):

    return state["route"]


# -----------------------------
# RAG Agent
# -----------------------------

def rag_agent(state):

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{state['question']}
"""

    response = llm.invoke(
        prompt
    )

    return {
        "context": context,
        "answer": response.content
    }


# -----------------------------
# Chat Agent
# -----------------------------

def chat_agent(state):

    response = llm.invoke(
        state["question"]
    )

    return {
        "answer": response.content
    }


# -----------------------------
# Graph
# -----------------------------

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

graph.set_entry_point(
    "supervisor"
)

graph.add_conditional_edges(
    "supervisor",
    route_decision,
    {
        "rag": "rag_agent",
        "chat": "chat_agent"
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

app = graph.compile()

# -----------------------------
# UI
# -----------------------------

question = st.text_input(
    "Ask Anything"
)

if question:

    result = app.invoke(
        {
            "question": question
        }
    )

    st.success(
        f"Route Used: {result['route'].upper()}"
    )

    st.markdown("## Answer")

    st.write(
        result["answer"]
    )