# SECTION 1: Install dependencies. Refer to .env.example to handle ENVIRONMENT VARIABLES
#  SECTION 2: IMPORTS AND CONFIG
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
import uuid
import Pinecone, ServerlessSpec
import pickle
from transformers import AutoTokenizer, AutoModel, pipeline, AutoModelForCausalLM
import openai
from huggingface_hub import login
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
import requests, gc, torch, numpy as np
from typing import TypedDict
from difflib import SequenceMatcher
from langgraph.graph import StateGraph, END

# add a debug wrapper for dev - for each node in LangGraph flow
def debug_node(name):
    def decorator(fn):
        def wrapper(state):
            print(f"\n🔧 Entering node: {name}")
            result = fn(state)
            print(f"✅ Exiting node: {name}")
            if isinstance(result, dict):
                try:
                    snapshot = {k: v for k, v in result.items() if k not in ["context", "final_answer", "validated_response"]}
                    print(f"🧠 State snapshot:\n{snapshot}")
                except Exception as e:
                    print(f"⚠️ Could not print state snapshot: {e}")
            else:
                print(f"🧠 Skipped snapshot — returned type is: {type(result)}")
            return result
        return wrapper
    return decorator


# SECTION 3: Authentication and PINECONE SETUP
# huuggingface_hub login
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# Pinecone set up
EMBEDDING_DIM = 3584
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "documentvectorstore")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

def init_pinecone_index(api_key: str, index_name: str):
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-west-1")
        )
    return pc.Index(index_name)

# Initialize at module level
index = init_pinecone_index(PINECONE_API_KEY, INDEX_NAME)

# ✅ SECTION 4: EMBEDDING PIPELINE (Qwen2-7B with Mean Pooling)
embed_pipe = pipeline(
    "feature-extraction",
    model="Alibaba-NLP/gte-Qwen2-7B-instruct",
    token=HF_API_TOKEN,
    device=-1  # CPU
)

def embed_text_batch(texts: list) -> np.ndarray:
    """
    Embeds a list of texts using Qwen2-7B with mean pooling.
    Ensures output shape is (batch_size, 3584).
    """
    if isinstance(texts, str):
        texts = [texts]

    outputs = embed_pipe(texts, padding=True, truncation=True, max_length=512)
    normalized_outputs = []
    for out in outputs:
        arr = np.array(out)
        while arr.ndim > 2:
            arr = arr[0]  # unwrap unnecessary batch dims
        normalized_outputs.append(np.mean(arr, axis=0))  # mean pooling

    return np.vstack(normalized_outputs)

def embed_text(text): return embed_text_batch([text])

# ✅ SECTION 5: RAG STATE DEFINITION
class RAGState(TypedDict):
    query: str
    expanded_query: str
    company: str
    year: str
    context: str
    swot_analysis: str
    validated_response: str
    response_type: str
    final_answer: str
    confidence: str

# ✅ SECTION 6: HUGGING FACE INFERENCE UTILITY
from huggingface_hub.inference._client import InferenceClient as HFInferenceClient

hf_llama_client = HFInferenceClient(model="meta-llama/Meta-Llama-3-8B-Instruct", token=HF_API_TOKEN)
hf_tiny_client = HFInferenceClient(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=HF_API_TOKEN)

def query_llm(prompt: str, max_tokens: int, model: str, temperature: float = 0.3) -> str:
    client = hf_llama_client if "llama" in model.lower() else hf_tiny_client
    return client.text_generation(prompt=prompt, max_new_tokens=max_tokens, temperature=temperature).strip()

# ✅  SECTION 7: LANGGRAPH FLOW DEFINIT
rag_graph = StateGraph(RAGState)

# === Node: expand_query_node ===
@debug_node("expand_query_node")
def expand_query_node(state: RAGState) -> RAGState:
    classify_prompt = f"Classify this query: {state['query']}\nType (swot, reasoning, simple):"
    try:
        query_type = query_llm(classify_prompt, max_tokens=5, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    except Exception:
        query_type = "simple"
    state["response_type"] = query_type.lower().strip() if query_type else "simple"

    prompt = f"""
    You are an AI assistant that expands search queries.
    Improve the following query by adding synonyms, related concepts, or clarifying terms, while preserving the original meaning.

    Query: {state['query']}
    Expanded Query:
    """
    try:
        expanded = query_llm(prompt, max_tokens=60, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        state["expanded_query"] = expanded.strip()
    except Exception as e:
        print(f"⚠️ Expansion failed: {e}")
        state["expanded_query"] = state["query"]

    print("🔍 Original Query:", state["query"])
    print("🧠 Expanded Query:", state["expanded_query"])
    print("📌 Detected Type:", state["response_type"])
    return state

# === Node: check_metadata ===
@debug_node("check_metadata")
def check_metadata(state: RAGState) -> RAGState:
    import re

    missing = []
    if not state.get("company"):
        missing.append("company")
    if not state.get("year"):
        missing.append("year")

    if missing:
        print("⚠️ Missing Metadata:", ', '.join(missing))
        clarification_prompt = "Please provide company name and year of the 10K annual report."

        try:
            clarifying_response = query_llm(clarification_prompt, max_tokens=50, model="meta-llama/Meta-Llama-3-8B-Instruct")
            print("🔁 Clarification Attempt:", clarifying_response)
            company_match = re.search(r"(?:company name is|company is|for)\s+([A-Z][\w&\-\s]{2,})", clarifying_response, re.IGNORECASE)
            year_match = re.search(r"(20\d{2})", clarifying_response)

            if company_match:
                state["company"] = company_match.group(1).strip()
            if year_match:
                state["year"] = year_match.group(1).strip()
        except Exception as e:
            print(f"⚠️ Clarification failed: {e}")

        if not state.get("company") or not state.get("year"):
            query = state["expanded_query"] or state["query"]
            web_prompt = f"The user asked: {query}. Since the company name or year of the 10-K is missing, summarize key insights based on general online search knowledge."

            try:
                answer = query_llm(web_prompt, max_tokens=300, model="meta-llama/Meta-Llama-3-8B-Instruct")
            except Exception as e:
                answer = f"Error during web-style response generation: {str(e)}"

            state["context"] = "[Web summary used due to missing metadata]"
            state["final_answer"] = answer
            state["validated_response"] = answer
            state["confidence"] = "medium"
            print("🌐 Used web search fallback")

    return state

# === Node: retrieve_context ===
@debug_node("retrieve_context")
def retrieve_context(state: RAGState) -> RAGState:
    context = state.get("context") or ""
    if "missing metadata" in context.lower():
        return state

    query_vector = embed_text(state["expanded_query"])[0].tolist()
    response = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        filter={"company": {"$eq": state["company"]}, "year": {"$eq": state["year"]}}
    )
    docs = [m["metadata"].get("chunk_text", "") for m in response.get("matches", []) if "metadata" in m]
    state["context"] = "\n\n".join(dict.fromkeys([d for d in docs if d.strip()])) or "[No relevant 10K context found]"
    return state

# === Node: determine_response_type ===
@debug_node("determine_response_type")
def determine_response_type(state: RAGState) -> RAGState:
    return state  # response_type already determined

# === Node: generate_response (and its branches) ===
@debug_node("generate_simple_answer")
def generate_simple_answer(state: RAGState) -> RAGState:
    prompt = f"""
    ### Role:
    You are a financial research assistant.

    ### Task:
    Provide a short, factual answer to the user query.

    ### Format:
    Respond in 1–2 very concise sentences.

    ### Context:
    {state['context']}

    ### Constraints:
    Do not explain or elaborate. Answer directly based on the provided context.

    ### Reference:
    Query: {state['query']}

    Answer:
    """
    state["final_answer"] = query_llm(prompt, max_tokens=100, model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return state

@debug_node("generate_reasoning_answer")
def generate_reasoning_answer(state: RAGState) -> RAGState:
    prompt = f"""
    ### Role:
    You are a strategy consultant supporting cross-functional product planning through analytical insights from the 10-K filing.

    ### Task:
    Provide a McKinsey-style response that evaluates the user query using structured reasoning and data-backed analysis.

    ### Format:
    Respond in 3–4 concise, executive-level sentences:
    1. Anchor in context with a key insight.
    2. Analyze the impact, drivers, or dependencies.
    3. Include one relevant metric or directional estimate.
    4. End with an actionable implication or recommendation.

    ### Context:
    {state['context']}

    ### Constraints:
    Maintain a structured, non-generic tone. Prioritize relevance, clarity, and strategic value.

    ### Reference:
    Question: {state['query']}

    Answer:
    """
    state["final_answer"] = query_llm(prompt, max_tokens=300, model="meta-llama/Meta-Llama-3-8B-Instruct")
    return state

@debug_node("generate_swot_answer")
def generate_swot_answer(state: RAGState) -> RAGState:
    prompt = f"""
    ### Role:
    You are a senior product strategy leader conducting a strategic review of the company’s 10-K filing.

    ### Task:
    Synthesize a SWOT analysis that helps prioritize product investments, mitigate delivery risk, and align product strategy with business outcomes.

    ### Format:
    - **Strengths**
    - **Weaknesses**
    - **Opportunities**
    - **Threats**
    - **Strategic Summary**

    ### Context:
    {state['context']}

    ### Constraints:
    Avoid boilerplate. Use data where possible. Highlight unique strategic insights.

    ### Reference:
    User question: {state['query']}

    SWOT Analysis:
    """
    state["final_answer"] = query_llm(prompt, max_tokens=700, model="meta-llama/Meta-Llama-3-8B-Instruct")
    return state


@debug_node("generate_response")
def generate_response(state: RAGState) -> RAGState:
    if state["response_type"] == "swot":
        return generate_swot_answer(state)
    elif state["response_type"] == "reasoning":
        return generate_reasoning_answer(state)
    else:
        return generate_simple_answer(state)

# === Node: validate_response ===
@debug_node("validate_response")
def validate_response(state: RAGState) -> RAGState:
    response_type = state.get("response_type", "simple")

    if response_type == "swot":
        prompt = f"""You are a product strategy consultant. Clean up this SWOT output and ensure it follows the structure (Strengths, Weaknesses, Opportunities, Threats, Strategic Summary):\n\n{state['final_answer']}"""
    elif response_type == "reasoning":
        prompt = f"""You are a strategy editor. Refine this response for clarity and logic:\n\n{state['final_answer']}"""
    else:
        prompt = f"""Tighten this short factual answer without changing meaning:\n\n{state['final_answer']}"""

    raw_response = query_llm(prompt, max_tokens=300, model="meta-llama/Meta-Llama-3-8B-Instruct")
    lines = list(dict.fromkeys(raw_response.splitlines()))
    state["validated_response"] = "\n".join(lines).strip()
    state["confidence"] = "high" if response_type != "simple" else "medium"
    return state

# === Add nodes and edges ===
rag_graph.add_node("expand_query", expand_query_node)
rag_graph.add_node("check_metadata", check_metadata)
rag_graph.add_node("retrieve_context", retrieve_context)
rag_graph.add_node("determine_response", determine_response_type)
rag_graph.add_node("generate_response", generate_response)
rag_graph.add_node("validate_response", validate_response)

rag_graph.add_edge("expand_query", "check_metadata")
rag_graph.add_edge("check_metadata", "retrieve_context")
rag_graph.add_edge("retrieve_context", "determine_response")
rag_graph.add_edge("determine_response", "generate_response")
rag_graph.add_edge("generate_response", "validate_response")
rag_graph.add_edge("validate_response", END)

rag_graph.set_entry_point("expand_query")
rag_pipeline = rag_graph.compile()
