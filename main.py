from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama


llm = ChatOllama(model="llama3.2")


class AgentState(TypedDict):
    question: str
    answer: str


def chatbot_node(state: AgentState):
    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }


graph = StateGraph(AgentState)

graph.add_node("chatbot", chatbot_node)

graph.set_entry_point("chatbot")

graph.add_edge("chatbot", END)

app = graph.compile()


result = app.invoke(
    {
        "question": "LangGraph kya hai?"
    }
)

print(result["answer"])