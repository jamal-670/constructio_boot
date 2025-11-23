import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import requests
import chromadb
import tempfile
import json
import time
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient, HfApi

st.set_page_config(page_title="NYC Construction Bot", page_icon="🏗️")

# --- DATABASE SETUP ---
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_data")

@st.cache_resource
def load_db():
    # 1. Connect to DB
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 2. THE FIX: Use get_or_create instead of just get
    # This prevents the "NotFoundError" crash
    collection = client.get_or_create_collection(name="construction_knowledge", embedding_function=ef)
    
    # 3. Check if empty. If empty, run the build script!
    if collection.count() == 0:
        print("⚠️ Database found but empty. Rebuilding...")
        import build_db
        # We run the builder manually
        build_db.build_database()
        # Re-connect to get the fresh data
        collection = client.get_collection(name="construction_knowledge", embedding_function=ef)
        
    return collection

# Load the database (This will now auto-fix itself if broken)
collection = load_db()

# --- LOGGING TO HUGGING FACE DATASET ---
def log_to_dataset(question, answer):
    try:
        token = st.secrets["HF_TOKEN"]
        api = HfApi(token=token)
        data = {
            "timestamp": time.time(),
            "instruction": question,
            "output": answer
        }
        filename = f"logs/{int(time.time())}.json"
        repo_id = "Ibrahimkhan2005/construction-bot-logs" 
        
        api.upload_file(
            path_or_fileobj=json.dumps(data).encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Log Error: {e}")

# --- API LOGIC ---
def get_ai_response(prompt):
    token = st.secrets["HF_TOKEN"] 
    client = InferenceClient(token=token)
    try:
        messages = [{"role": "user", "content": prompt}]
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages, 
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- NOTIFICATIONS ---
def send_discord(name, phone, details):
    url = st.secrets["DISCORD_WEBHOOK_URL"]
    data = {"content": f"🚀 **Lead:** {name} | {phone}\n{details}"}
    requests.post(url, json=data)

# --- UI ---
st.title("🏗️ G5 Construction Estimator")

if "registered" not in st.session_state: st.session_state.registered = False
if "messages" not in st.session_state: st.session_state.messages = [{"role": "assistant", "content": "Hello! Register to start."}]

if not st.session_state.registered:
    with st.form("reg"):
        name = st.text_input("Name")
        phone = st.text_input("Phone")
        desc = st.text_area("Project")
        if st.form_submit_button("Start"):
            if name and phone:
                send_discord(name, phone, desc)
                st.session_state.registered = True
                st.rerun()
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # RAG Logic
        context = ""
        results = collection.query(query_texts=[prompt], n_results=3)
        if results['documents']: context = "\n".join(results['documents'][0])
        
        final_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer as expert:"
        resp = get_ai_response(final_prompt)
        
        with st.chat_message("assistant"): st.write(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        
        log_to_dataset(prompt, resp)
