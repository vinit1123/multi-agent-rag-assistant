import streamlit as st
import requests

st.set_page_config(
    page_title="AI Assistant",
    layout="wide"
)

st.title("🤖 AI Assistant")

# -----------------------------
# Chat History
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(
        msg["role"]
    ):
        st.markdown(
            msg["content"]
        )

# -----------------------------
# User Input
# -----------------------------

question = st.chat_input(
    "Ask Anything..."
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

    # -----------------------------
    # Build History
    # -----------------------------

    history = ""

    for msg in st.session_state.messages:

        history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    # -----------------------------
    # API Call
    # -----------------------------

    with st.spinner("Thinking..."):

        response = requests.post(
            "http://127.0.0.1:8000/chat",
            json={
                "question": history
            }
        )

        result = response.json()

    answer = result["answer"]

    route = result["route"]

    # -----------------------------
    # Assistant Response
    # -----------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        st.caption(
            f"Route: {route.upper()}"
        )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )