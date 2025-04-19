import os
import streamlit as st
from prompts import PromptTemplates
from pipeline import rag_pipeline

# === App Config ===
st.set_page_config(page_title="10-K RAG LLM Chatbot", layout="wide")
st.title("💬 10-K Company Insight AI Assistant")
st.markdown("Ask simple or strategic questions based on a company's 10-K filing.")

# === Sidebar Controls ===
st.sidebar.header("Settings")
company = st.sidebar.selectbox("🏢 Select Company", ["Workday", "Salesforce", "Oracle", "ADP", "SAP", "SAP", "nvidia", "Amazon", "Google", "Paypal"])
year = st.sidebar.selectbox("📅 Select Year", ["2024"])
debug_toggle = st.sidebar.checkbox("Enable Debug Mode", value=False)
os.environ["DEBUG_MODE"] = "true" if debug_toggle else "false"

# === Initialize Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# === User Chat Input ===
query = st.chat_input("Ask a question about the company's 10-K report")
if query:
    with st.spinner("🔄 Running the RAG pipeline..."):
        result = rag_pipeline.invoke({
            "query": query,
            "company": company,
            "year": year
        })

    # Append to history
    st.session_state.chat_history.append({
        "user": query,
        "answer": result["final_answer"],
        "references": result.get("references", []),
        "debug_info": result.get("debug_info") if debug_toggle else None
    })

# === Display Chat ===
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(msg["user"])
    with st.chat_message("assistant"):
        st.markdown(msg["answer"])
        if msg.get("references"):
            with st.expander("📚 References"):
                for ref in msg["references"]:
                    st.markdown(f"- **Page {ref['page']}** from `{ref['pdf_path']}`")
        if msg.get("debug_info"):
            with st.expander("🛠️ Debug Info"):
                st.code(msg["debug_info"])
