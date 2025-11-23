import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import chromadb
import tempfile
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

st.set_page_config(page_title="NYC Construction Bot", page_icon="🏗️")

# --- DATABASE SETUP ---
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_data")

@st.cache_resource
def load_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="construction_knowledge", embedding_function=ef)
    
    # FORCE REBUILD if empty (Self-Healing)
    if collection.count() == 0:
        import build_db
        build_db.build_database()
        collection = client.get_or_create_collection(name="construction_knowledge", embedding_function=ef)
        
    return collection

collection = load_db()

# --- API LOGIC (STRICT MODE) ---
def get_ai_response(prompt, context_data):
    token = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=token)
    
    # SYSTEM PROMPT: Forces the AI to use your data
    system_message = (
        "You are a professional Estimator for BuildSmart NYC. "
        "Your job is to provide price estimates based ONLY on the Context Data provided below. "
        "Rules:\n"
        "1. If the context contains a price range (e.g., $4.50 - $7.50), USE THAT EXACT RANGE.\n"
        "2. Do not use outside knowledge for prices.\n"
        "3. If the answer is not in the context, say 'I don't have that specific pricing available.'\n"
        "4. Format the answer with bullet points."
    )
    
    # Combine User Prompt + Database Data
    full_prompt = f"Context Data:\n{context_data}\n\nUser Question: {prompt}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": full_prompt}
    ]
    
    try:
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages, 
            max_tokens=500,
            temperature=0.1 # Low temperature = Stick to facts
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- UI ---
st.title("🏗️ NYC Construction Estimator")

if "messages" not in st.session_state: 
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me about roofing, masonry, or kitchen prices."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.write(prompt)
    
    # 1. RETRIEVE DATA
    context = ""
    # We fetch 3 results to ensure we get the right price
    results = collection.query(query_texts=[prompt], n_results=3)
    
    if results['documents']: 
        context = "\n".join(results['documents'][0])
    
    # 2. DEBUG BOX (Check this to see if DB is working!)
    with st.expander("🔍 Debug: What the AI read from Database"):
        if context:
            st.info(context)
        else:
            st.error("No data found! Database might be empty.")

    # 3. GENERATE ANSWER
    with st.chat_message("assistant"):
        with st.spinner("Calculating Estimate..."):
            resp = get_ai_response(prompt, context)
            st.write(resp)
            
    st.session_state.messages.append({"role": "assistant", "content": resp})
