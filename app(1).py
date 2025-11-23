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
if not os.path.exists(CHROMA_DB_PATH):
    import build_db
    build_db.build_database()

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    return client.get_collection(name="construction_knowledge", embedding_function=ef)

collection = load_db()

# --- LOGGING TO HUGGING FACE DATASET ---
def log_to_dataset(question, answer):
    """
    Saves the Q&A pair to your Hugging Face Dataset for future training.
    """
    try:
        # Get Token from Secrets
        token = st.secrets["HF_TOKEN"]
        api = HfApi(token=token)
        
        # Prepare Data (JSON format is best for training)
        data = {
            "timestamp": time.time(),
            "instruction": question,
            "output": answer
        }
        
        # Create a unique filename so we don't overwrite old logs
        filename = f"logs/{int(time.time())}.json"
        
        # Upload to your dataset
        # CHANGE 'Ibrahimkhan2005' TO YOUR USERNAME IF DIFFERENT
        repo_id = "Ibrahimkhan2005/construction-bot-logs" 
        
        api.upload_file(
            path_or_fileobj=json.dumps(data).encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Data saved to Hugging Face Dataset")
    except Exception as e:
        print(f"⚠️ Could not save log: {e}")

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
st.title("🏗️ NYC Construction Estimator")

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
        # 1. Show User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # 2. RAG Logic
        context = ""
        results = collection.query(query_texts=[prompt], n_results=3)
        if results['documents']: context = "\n".join(results['documents'][0])
        
        final_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer as expert:"
        
        # 3. Get AI Response
        resp = get_ai_response(final_prompt)
        
        # 4. Show AI Response
        with st.chat_message("assistant"): st.write(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        
        # 5. SAVE DATA FOR TRAINING (The New Part)
        log_to_dataset(prompt, resp)
