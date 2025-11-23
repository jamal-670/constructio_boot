# --- 1. SQLITE FIX (MUST BE AT THE VERY TOP) ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------------

import streamlit as st
import os
import chromadb
import tempfile
import shutil
import time
import json
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient, HfApi

# --- CONFIGURATION ---
st.set_page_config(page_title="NYC Construction Bot", page_icon="🏗️")

# --- KNOWLEDGE BASE (DATA) ---
# We put the data right here to avoid import errors
DATA_DOCUMENTS = [
    # ROOFING
    "SERVICE: Asphalt Shingles Roofing. PRICE: $4.50 – $7.50 per sq ft. NOTES: Most common residential roof.",
    "SERVICE: Flat Roofing (TPO/EPDM). PRICE: $6.00 – $11.00 per sq ft. NOTES: Commercial/low-slope.",
    "SERVICE: Roof Repair. PRICE: $150 – $600 per job. NOTES: Minor patching.",
    
    # MASONRY & CONCRETE
    "SERVICE: Brick Veneer. PRICE: $12.00 – $25.00 per sq ft. NOTES: Face brick installation.",
    "SERVICE: Concrete Slab. PRICE: $4.50 – $9.50 per sq ft. NOTES: Pour and finish.",
    "SERVICE: Concrete Sidewalk. PRICE: $6.00 – $11.00 per linear ft.",
    
    # INTERIOR
    "SERVICE: Kitchen Renovation (Small). PRICE: $22,000 – $45,000. NOTES: Basic finishes.",
    "SERVICE: Kitchen Renovation (Medium). PRICE: $40,000 – $110,000. NOTES: Mid-range finishes.",
    "SERVICE: Bathroom Renovation. PRICE: $15,000 – $35,000 (Average).",
    
    # COMPANY INFO
    "COMPANY: BuildSmart NYC. PHONE: (212) 555-0199. EMAIL: contact@buildsmartnyc.com.",
    "COMPANY: BuildSmart NYC. ADDRESS: 350 5th Ave, New York, NY 10118.",
    "POLICY: Warranty. We offer a 5-year workmanship warranty on structural work."
]
DATA_IDS = [str(i) for i in range(len(DATA_DOCUMENTS))]

# --- DATABASE SETUP ---
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_data")

@st.cache_resource
def get_collection():
    """
    Handles Database Creation and Loading in one function.
    """
    # 1. Initialize Client
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 2. Get or Create Collection
    collection = client.get_or_create_collection(name="construction_knowledge", embedding_function=ef)
    
    # 3. SELF-HEALING: If empty, fill it with data!
    if collection.count() == 0:
        print("⚠️ DB Empty. Building now...")
        collection.add(documents=DATA_DOCUMENTS, ids=DATA_IDS)
        print("✅ DB Built successfully.")
        
    return collection

# Load DB (This runs once when app starts)
collection = get_collection()

# --- API LOGIC ---
def get_ai_response(prompt, context_data):
    token = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=token)
    
    system_message = (
        "You are a professional Estimator for BuildSmart NYC. "
        "Your job is to provide price estimates based ONLY on the Context Data provided below. "
        "Rules:\n"
        "1. If the context contains a price range, USE THAT EXACT RANGE.\n"
        "2. Do not use outside knowledge for prices.\n"
        "3. If the answer is not in the context, say 'I don't have that specific pricing available.'\n"
        "4. Format the answer with bullet points."
    )
    
    full_prompt = f"Context Data:\n{context_data}\n\nUser Question: {prompt}"
    
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages, 
            max_tokens=500,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- LOGGING ---
def log_to_dataset(question, answer):
    try:
        token = st.secrets["HF_TOKEN"]
        api = HfApi(token=token)
        data = {"timestamp": time.time(), "question": question, "answer": answer}
        # CHANGE USERNAME IF NEEDED
        repo_id = "Ibrahimkhan2005/construction-bot-logs"
        api.upload_file(
            path_or_fileobj=json.dumps(data).encode("utf-8"),
            path_in_repo=f"logs/{int(time.time())}.json",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except:
        pass

# --- UI ---
st.title("🏗️ NYC Construction Estimator")

if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about roofing, masonry, or kitchen prices."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    # 1. RETRIEVE
    context = ""
    results = collection.query(query_texts=[prompt], n_results=3)
    if results['documents']: context = "\n".join(results['documents'][0])
    
    # 2. DEBUG BOX
    with st.expander("🔍 Debug: Database Context"):
        if context: st.info(context)
        else: st.error("No matching data found.")

    # 3. GENERATE
    with st.chat_message("assistant"):
        with st.spinner("Estimating..."):
            resp = get_ai_response(prompt, context)
            st.write(resp)
            
    st.session_state.messages.append({"role": "assistant", "content": resp})
    log_to_dataset(prompt, resp)
