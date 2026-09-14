"""
AI 410 — frontend chat client (Streamlit).

Talks to the local FastAPI backend over HTTP, server-side — this runs
as its own Python process, not JavaScript in the browser, so there's
no CORS involved. Change API_BASE if you deploy the backend elsewhere
(see the Week 10 guide).
"""

import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Chat")
st.title("Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask something…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            res = requests.post(f"{API_BASE}/chat", json={"message": prompt}, timeout=30)
            res.raise_for_status()
            reply = res.json()["reply"]
        except requests.RequestException as e:
            reply = f"⚠️ Error talking to the backend: {e}"
        st.write(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
