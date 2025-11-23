# --- 1. SQLITE FIX (MUST BE AT THE VERY TOP) ---
# This fixes the "Read-only database" crash on Hugging Face
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------------

import streamlit as st
import os
import requests
import chromadb
import tempfile
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient

# --- Configuration & Styling ---
st.set_page_config(
    page_title="NYC Construction Assistant",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Database Loading (Smart Path) ---

try:
    with open("test_permissions.txt", "w") as f:
        f.write("test")
    os.remove("test_permissions.txt")
    CHROMA_DB_PATH = os.path.abspath("./chroma_db_data")
except Exception:
    CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_data")

COLLECTION_NAME = "construction_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Build DB if missing
if not os.path.exists(CHROMA_DB_PATH):
    if os.path.exists("build_db.py"):
        import build_db
        with st.spinner("Building database..."):
            build_db.build_database()

@st.cache_resource
def load_db():
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        query_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=query_ef)
        return collection
    except Exception as e:
        print(f"❌ ChromaDB Error: {e}")
        return None

with st.spinner("🏗️ Connecting to Construction Data..."):
    collection = load_db()

# --- 3. API Model Logic (Hugging Face Official) ---

def query_huggingface_api(prompt):
    # Get Token
    token = os.environ.get("HF_TOKEN")
    if not token:
        return "⚠️ Error: HF_TOKEN is missing in Settings -> Secrets."

    # Setup Client (Defaults to Hugging Face URL)
    client = InferenceClient(token=token)

    try:
        messages = [
            {"role": "system", "content": "You are a professional NYC construction estimator. Answer strictly based on the Context provided. Use bullet points for prices."},
            {"role": "user", "content": prompt}
        ]
        
        # Using Llama-3-8B-Instruct
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages, 
            max_tokens=500, 
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error contacting AI Brain: {e}"

# --- 4. Notification Logic ---

def send_discord_notification(client_name, client_phone, client_email, details):
    webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook_url: return False

    data = {
        "embeds": [{
            "title": "🚀 New Construction Lead!",
            "color": 3066993,
            "fields": [
                {"name": "👤 Name", "value": client_name, "inline": True},
                {"name": "📞 Phone", "value": client_phone, "inline": True},
                {"name": "📧 Email", "value": client_email, "inline": False},
                {"name": "📝 Project", "value": details, "inline": False}
            ],
            "footer": {"text": "Sent from G5 Construction Bot"}
        }]
    }
    try:
        requests.post(webhook_url, json=data)
        return True
    except:
        return False

# --- 5. RAG Logic ---

def generate_rag_response(query):
    retrieved_context = "No specific data found."
    if collection:
        try:
            results = collection.query(query_texts=[query], n_results=3)
            if results['documents']:
                retrieved_context = "\n".join(results['documents'][0])
        except:
            pass

    full_prompt = f"""
    Context Data from Database:
    {retrieved_context}
    
    User Question: {query}
    
    Please provide a professional estimate based on the Context Data above.
    """
    return query_huggingface_api(full_prompt)

# --- 6. UI Logic ---

if "is_registered" not in st.session_state:
    st.session_state.is_registered = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Please fill out your details to start chatting."}]

if not st.session_state.is_registered:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=150)
        st.title("🏗️ Welcome")
        st.markdown("Please sign in to consult with our AI Estimator.")
        
        with st.form("reg_form"):
            name = st.text_input("Full Name")
            phone = st.text_input("Phone Number")
            email = st.text_input("Email Address")
            project_desc = st.text_area("Brief Project Description")
            submitted = st.form_submit_button("Start Chatting")
            
            if submitted:
                if name and phone:
                    st.success("Registered!")
                    send_discord_notification(name, phone, email, project_desc)
                    st.session_state.is_registered = True
                    st.session_state.messages = [{"role": "assistant", "content": f"Hi {name}! How can I help you with your construction project today?"}]
                    st.rerun()
                else:
                    st.error("Please enter at least your Name and Phone.")
else:
    st.title("🏗️ Construction Assistant")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask about pricing, materials, or codes..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_text = generate_rag_response(user_input)
                st.write(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})