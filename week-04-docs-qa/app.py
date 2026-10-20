"""Streamlit UI. Talks to the FastAPI service.

    streamlit run app.py
"""
import json
import urllib.request

import streamlit as st

API = "http://localhost:8000/ask"

st.set_page_config(page_title="Handbook Q&A", page_icon="📘")
st.title("📘 Northwind handbook Q&A")
st.caption("Answers come only from the handbook, with page citations.")

EXAMPLES = [
    "What is the deadline to submit an expense claim?",
    "How much is the standby allowance for the primary on-call engineer?",
    "Who has to approve an expense claim of INR 60,000?",
    "What is the stock option vesting schedule?",
]
pick = st.selectbox("Try an example", [""] + EXAMPLES)
question = st.text_input("Or ask your own question", value=pick)
k = st.slider("Chunks to retrieve (k)", 1, 10, 5)

if st.button("Ask", type="primary") and question.strip():
    body = json.dumps({"question": question, "k": k}).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    try:
        with st.spinner("Searching and thinking..."):
            with urllib.request.urlopen(req, timeout=300) as r:
                result = json.load(r)
    except Exception as e:
        st.error(f"Could not reach the API at {API}. Is `uvicorn api:app` running?\n\n{e}")
        st.stop()

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    for s in result["sources"]:
        with st.expander(f"[{s['tag']}]  page {s['page']}  ·  similarity {s['score']}"):
            st.text(s["text"])