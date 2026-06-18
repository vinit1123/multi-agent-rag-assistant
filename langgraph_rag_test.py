from typing import TypedDict

from langgraph.graph import StateGraph, END

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama


# --------------------
# Build Vector Store
# --------------------

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

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOllama(
    model="llama3.2"
)


# --------------------
# State
# --------------------

class AgentState(TypedDict):
    question: str
    context: str
    answer: str


# --------------------
# Retrieve Node
# --------------------

def retrieve_node(state: AgentState):

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


# --------------------
# Generate Node
# --------------------

def generate_node(state: AgentState):

    prompt = f"""
Answer using ONLY the context below.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# --------------------
# Graph
# --------------------

graph = StateGraph(AgentState)

graph.add_node(
    "retrieve",
    retrieve_node
)

graph.add_node(
    "generate",
    generate_node
)

graph.set_entry_point(
    "retrieve"
)

graph.add_edge(
    "retrieve",
    "generate"
)

graph.add_edge(
    "generate",
    END
)

app = graph.compile()


# --------------------
# Run
# --------------------

result = app.invoke(
    {
        "question":
        "What is Multi Head Attention?"
    }
)

print()
print("ANSWER:")
print(result["answer"])