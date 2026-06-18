from fastapi import FastAPI
from pydantic import BaseModel

from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_ollama import (
    ChatOllama,
    OllamaEmbeddings
)

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

# -----------------------------------
# FastAPI
# -----------------------------------

app = FastAPI()

# -----------------------------------
# LLM
# -----------------------------------

llm = ChatOllama(
    model="llama3.2"
)

# -----------------------------------
# Build RAG
# -----------------------------------

loader = PyPDFLoader(
    "data/sample.pdf"
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(
    docs
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------------
# Request Schema
# -----------------------------------

class ChatRequest(BaseModel):
    question: str

# -----------------------------------
# LangGraph State
# -----------------------------------

class AgentState(TypedDict):
    question: str
    route: str
    context: str
    answer: str

# -----------------------------------
# Supervisor
# -----------------------------------
def supervisor(state):

    import re

    q = state["question"].lower()

    # Latest user question nikalo
    lines = q.strip().split("\n")

    latest_question = lines[-1]

    if ":" in latest_question:
        latest_question = (
            latest_question
            .split(":", 1)[1]
            .strip()
        )

    # Tool Route

    if re.fullmatch(
        r"\s*[0-9\+\-\*\/\.\(\)\s]+\s*",
        latest_question
    ):
        return {"route": "tool"}

    # RAG Route

    pdf_keywords = [
        "attention",
        "transformer",
        "encoder",
        "decoder",
        "multi head"
    ]

    if any(
        word in latest_question
        for word in pdf_keywords
    ):
        return {"route": "rag"}

    # Chat Route

    return {"route": "chat"}

# -----------------------------------
# Router
# -----------------------------------

def router(state):

    return state["route"]

# -----------------------------------
# Chat Agent
# -----------------------------------

def chat_agent(state):

    prompt = f"""
You are a helpful AI assistant.

The text below contains the complete conversation history.

Use previous messages when answering.

Conversation History:

{state["question"]}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }
# -----------------------------------
# Tool Agent
# -----------------------------------

def tool_agent(state):

    import re

    try:

        q = state["question"]

        matches = re.findall(
            r"\d+(?:\.\d+)?|[\+\-\*\/\(\)]",
            q
        )

        expression = "".join(matches)

        if not expression:
            return {
                "answer": "Invalid calculation"
            }

        result = eval(expression)

        return {
            "answer": str(result)
        }

    except Exception as e:

        return {
            "answer": f"Invalid calculation"
        }
# -----------------------------------
# RAG Agent
# -----------------------------------
def rag_agent(state):

    print("\nQUESTION RECEIVED:")
    print(state["question"])
    print("\n")

    docs = retriever.invoke(
        state["question"]
    )

    docs = retriever.invoke(
        state["question"]
    )

    print("\n===================")

    for i, doc in enumerate(docs):

        print(f"\nChunk {i+1}\n")

        print(
            doc.page_content[:500]
        )

    print("\n===================")

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    prompt = f"""
You are a document assistant.

Use ONLY the supplied context.

If the answer is not present in the context, say:

I could not find that information in the uploaded PDFs.

Context:
{context}

Conversation:
{state["question"]}

Answer:
"""

    response = llm.invoke(
        prompt
    )

    return {
        "context": context,
        "answer": response.content
    }

# -----------------------------------
# Build Graph
# -----------------------------------

graph = StateGraph(
    AgentState
)

graph.add_node(
    "supervisor",
    supervisor
)

graph.add_node(
    "chat_agent",
    chat_agent
)

graph.add_node(
    "tool_agent",
    tool_agent
)

graph.add_node(
    "rag_agent",
    rag_agent
)

graph.set_entry_point(
    "supervisor"
)

graph.add_conditional_edges(
    "supervisor",
    router,
    {
        "chat": "chat_agent",
        "tool": "tool_agent",
        "rag": "rag_agent"
    }
)

graph.add_edge(
    "chat_agent",
    END
)

graph.add_edge(
    "tool_agent",
    END
)

graph.add_edge(
    "rag_agent",
    END
)

agent = graph.compile()

# -----------------------------------
# API Endpoint
# -----------------------------------

@app.post("/chat")
def chat(req: ChatRequest):

    result = agent.invoke(
        {
            "question": req.question
        }
    )

    return {
        "route": result["route"],
        "answer": result["answer"]
    }