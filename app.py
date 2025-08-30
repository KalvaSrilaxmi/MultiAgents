import streamlit as st
from components import MasterAgent
import os

st.set_page_config(page_title="LangChain Multi-Agent Assistant", layout="wide")

# Load secrets
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") or st.secrets.get("WEATHER_API_KEY", "")

if not HF_TOKEN or not WEATHER_API_KEY:
    st.error("Please set HF_TOKEN and WEATHER_API_KEY in your environment or Streamlit secrets.")
    st.stop()

agent = MasterAgent(HF_TOKEN, WEATHER_API_KEY)

if "history" not in st.session_state:
    st.session_state.history = []

st.title("🤖 LangChain Multi-Agent Assistant")

# Chat display
for entry in st.session_state.history:
    if entry["role"] == "user":
        st.markdown(f"🧑 **You:** {entry['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {entry['content']}")

# Input
query = st.text_input("Type your message...", key="input")
if st.button("Send") or (query and st.session_state.get("input_submitted", False)):
    if query:
        st.session_state.history.append({"role": "user", "content": query})
        response = agent.route(query)
        st.session_state.history.append({"role": "bot", "content": response})
        st.rerun()

# Clear chat
if st.button("Clear Chat"):
    st.session_state.history = []
    st.rerun()

# Export conversation
def export_convo():
    convo = ""
    for entry in st.session_state.history:
        prefix = "You: " if entry["role"] == "user" else "Bot: "
        convo += f"{prefix}{entry['content']}\n"
    return convo

st.download_button("Export Conversation", export_convo(), "conversation.txt")

st.markdown("---")
st.markdown("Built with LangChain, Hugging Face, DuckDuckGo,")