import streamlit as st
import os
import chromadb
import tempfile
import time
import json
import requests
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient, HfApi

# --- CONFIGURATION ---
st.set_page_config(page_title="G5 Construction NYC Estimator", page_icon="🏗️", layout="wide")

# --- KNOWLEDGE BASE (THE UNIFIED MASTER LIST) ---
DATA_DOCUMENTS = [
    # ==========================================
    # 1. GENERAL CONDITIONS (The "Hidden" NYC Costs)
    # ==========================================
    "SERVICE: General Conditions & Logistics. LOCATION: NYC. COST: 10% – 20% of Project Value. NOTES: Covers site protection (Masonite for floors), dust barriers, project management, and daily cleanup. Required for Co-op/Condo compliance.",
    "SERVICE: DOB Permits & Filing. LOCATION: NYC. COST: $3,500 – $8,000+ per trade (General, Plumbing, Electric). NOTES: Required for moving walls, gas work, or structural changes. Architect/Expediter fees are separate.",
    "SERVICE: Debris Removal (Live Load). LOCATION: NYC. COST: $800 – $1,500 per truckload. NOTES: In Manhattan/areas where dumpsters aren't allowed, we use 'Live Load' (truck waits while we load). Includes disposal fees.",
    "SERVICE: Interior Protection & Board Approval Prep. LOCATION: NYC. COST: $1,500 – $3,500. NOTES: Includes submitting COI (Certificate of Insurance), protecting common hallways/elevators, and adhering to building work hours (usually 9AM–4PM).",

    # ==========================================
    # 2. ROOFING: SLOPED (Residences in Queens/Brooklyn/Staten Island)
    # ==========================================
    "SERVICE: Asphalt Shingles (Architectural/Laminated). LOCATION: NYC. PRICE: $5.00 – $8.00 per sq ft. SPECS: Fiberglass mat, Class A Fire Rating. WEIGHT: Moderate (2-4 lbs/sq ft). PROS: Cost-effective, 20-30 year lifespan, wind resistant up to 110mph. CONS: Vulnerable to ice dams (requires Ice & Water shield). BEST FOR: Standard pitched roofs.",
    "SERVICE: Slate Roofing (Natural Stone). LOCATION: NYC. PRICE: $35.00 – $65.00+ per sq ft. SPECS: Natural quarried stone, Non-combustible. WEIGHT: Very Heavy (8-12+ lbs/sq ft - Structural reinforcement often needed). LIFESPAN: 75–100+ years. NYC NOTES: Often required by Landmarks Preservation Commission (LPC) in historic districts.",
    "SERVICE: Metal Roofing (Standing Seam). LOCATION: NYC. PRICE: $14.00 – $22.00 per sq ft. SPECS: Galvalume or Aluminum with concealed fasteners. WEIGHT: Light (1-2 lbs/sq ft). LIFESPAN: 40–70 years. PROS: Excellent snow shedding, high wind uplift resistance. NYC NOTES: Snow guards are mandatory to protect pedestrians.",
    "SERVICE: Synthetic/Composite Slate. LOCATION: NYC. PRICE: $12.00 – $20.00 per sq ft. SPECS: Polymer/Rubber composite. WEIGHT: Moderate. PROS: Mimics slate look but lighter and cheaper. LIFESPAN: 50 years.",

    # ==========================================
    # 3. ROOFING: FLAT (Brownstones, Commercial, Multi-family)
    # ==========================================
    "SERVICE: Flat Roofing (SBS Modified Bitumen / Torch-down). LOCATION: NYC. PRICE: $8.50 – $14.00 per sq ft. SPECS: Multi-ply asphalt sheets. PROS: Tough, durable (20-30 years), handles foot traffic well. CONS: Dark color absorbs heat. MAINTENANCE: Check seams and flashings annually.",
    "SERVICE: Flat Roofing (TPO - Thermoplastic). LOCATION: NYC. PRICE: $9.00 – $15.00 per sq ft. SPECS: Single-ply white membrane, heat-welded seams. PROS: 'Cool Roof' (High reflectivity reduces summer AC costs), resists ponding water better than rubber. LIFESPAN: 20-25 years. NYC NOTES: Best for energy code compliance.",
    "SERVICE: Flat Roofing (EPDM Rubber). LOCATION: NYC. PRICE: $8.00 – $13.00 per sq ft. SPECS: Synthetic black rubber (glue/tape seams). PROS: Very flexible in freeze/thaw cycles. CONS: Black color gets hot, seams can fail if not taped perfectly. LIFESPAN: 20-30 years.",
    "SERVICE: Liquid Applied Roofing (Kemper/Siplast). LOCATION: NYC. PRICE: $25.00 – $45.00 per sq ft. SPECS: Resin-based, seamless waterproofing. PROS: Bulletproof, no seams to leak, self-flashing. BEST FOR: High-end terraces with complicated penetrations or paver systems.",

    # ==========================================
    # 4. KITCHENS (NYC Specific Scenarios)
    # ==========================================
    "SERVICE: Full Kitchen Renovation (Economy/Rental Grade). LOCATION: NYC. TOTAL: $25,000 – $40,000. SCOPE: Ikea/Stock cabinets, laminate or basic quartz, vinyl floor, keeping layout same. TIMELINE: 3–5 weeks.",
    "SERVICE: Full Kitchen Renovation (Mid-Range/Co-op). LOCATION: NYC. TOTAL: $45,000 – $85,000. SCOPE: Semi-custom cabinets, stone countertops, new lighting, tile backsplash, appliance install. TIMELINE: 5–8 weeks.",
    "SERVICE: Full Kitchen Renovation (High-End/Luxury). LOCATION: NYC. TOTAL: $90,000 – $200,000+. SCOPE: Custom millwork/inset cabinets, movement of plumbing/gas (requires permits), high-end appliances (SubZero/Wolf). TIMELINE: 8–14 weeks.",
    "SERVICE: Kitchen Logistics (NYC Wet over Dry). NOTE: Most Co-op boards forbid moving 'wet' areas (sinks) over 'dry' areas (bedrooms) in the apartment below. This limits layout changes.",
    "SERVICE: Gas Line Work. NOTE: Any work on gas lines in NYC requires a pressure test which may trigger a building-wide gas shutdown. We often recommend switching to Electric Induction cooktops to avoid this.",

    # ==========================================
    # 5. BATHROOMS
    # ==========================================
    "SERVICE: Full Bathroom Gut Renovation (5x8 Standard). LOCATION: NYC. TOTAL: $22,000 – $45,000. SCOPE: Demo to studs, new cement board, waterproofing, tiling to ceiling, new fixtures. TIMELINE: 3–6 weeks.",
    "SERVICE: Luxury Master Bath Renovation. LOCATION: NYC. TOTAL: $50,000 – $100,000+. SCOPE: Double vanity, walk-in shower with body sprays, freestanding tub, wall-hung toilet (requires carrier work), high-end stone. TIMELINE: 6–10 weeks.",
    "SERVICE: Tub-to-Shower Conversion. LOCATION: NYC. TOTAL: $12,000 – $22,000. SCOPE: Remove tub, install shower base, waterproofing up to 6ft, glass door install.",

    # ==========================================
    # 6. MASONRY, FACADE & INTERIORS
    # ==========================================
    "SERVICE: Brick Pointing / Tuckpointing. LOCATION: NYC. UNIT PRICE: $6.00 – $15.00 per sq ft. SCOPE: Grinding out joints and refilling. NOTE: Scaffolding/Sidewalk Shed is EXTRA (approx $4,000 - $10,000 for shed) and required for heights over 40ft.",
    "SERVICE: Brownstone Facade Restoration. LOCATION: NYC. TOTAL: $40,000 – $150,000 per facade. SCOPE: Scrape, patch, and re-coat brownstone mix. Highly skilled labor required.",
    "SERVICE: Interior Painting (Level 5 Skim Coat). LOCATION: NYC. UNIT PRICE: $6.00 – $12.00 per sq ft of wall. SCOPE: Plastering entire wall smooth (no texture) before painting. Standard for pre-war luxury renovations.",
    "SERVICE: New Engineered Wood Floor Install. LOCATION: NYC. UNIT PRICE: $10.00 – $25.00 per sq ft (Labor + Material). NOTE: Glue-down over soundproofing mat (cork/rubber) is standard for condos to meet STC noise ratings.",

    # ==========================================
    # 7. SPECIALTY (Green/Solar)
    # ==========================================
    "SERVICE: Green Roof (Extensive/Sedum). LOCATION: NYC. PRICE: $30.00 – $70.00 per sq ft. WEIGHT: Adds 15–30 lbs/sq ft (wet). NOTE: Requires structural engineering report.",
    "SERVICE: Solar-Integrated Roofing. LOCATION: NYC. PRICE: Varies by KW system size. NOTE: Roof substrate must be new. We coordinate with electricians.",

    # ==========================================
    # 8. COMPANY INFO
    # ==========================================
    "COMPANY: G5 Construction. PHONE: (212) 555-0199. EMAIL: Info@g5construction.net.",
    "COMPANY: Warranty. 5-Year Workmanship on residential, 15-20 Year NDL on Commercial Flat roofs.",
    "COMPANY: Service Area. Manhattan, Brooklyn, Queens, Bronx."
]

DATA_IDS = [str(i) for i in range(len(DATA_DOCUMENTS))]

# --- DATABASE SETUP ---
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_g5_unified")

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    # Force delete old DB to ensure new unified data is loaded
    try: client.delete_collection("g5_data_unified")
    except: pass
    collection = client.create_collection(name="g5_data_unified", embedding_function=ef)
    collection.add(documents=DATA_DOCUMENTS, ids=DATA_IDS)
    return collection

collection = get_collection()

# --- AI LOGIC (Unified Project Manager) ---
def get_ai_response(prompt, context_data):
    token = st.secrets["HF_TOKEN"]
    client = InferenceClient(token=token)
    
    system_message = (
        "You are the Senior Project Manager for G5 Construction in NYC. "
        "You handle everything from brownstone roofing to luxury apartment renovations."
        "\n\n"
        "**YOUR GUIDELINES:**\n"
        "1. **Context is King:** Use the provided data. If the user asks about Kitchens, use the 'Economy vs Luxury' ranges. If Roofing, use the 'Technical Specs'.\n"
        "2. **The NYC Reality Check:**\n"
        "   - If they mention Gas work, warn about 'Gas Tests/Induction'.\n"
        "   - If they mention moving a kitchen, mention 'Wet over Dry' rules.\n"
        "   - If they mention Slate/Green Roof, warn about 'Weight/Structure'.\n"
        "3. **Hidden Costs:** Always remind them that prices listed usually exclude **Permits** and **General Conditions (10-20%)** unless stated otherwise.\n"
        "4. **Be Consultative:** Ask for the property type (Co-op vs House) and size to give a better estimate.\n"
        "5. **Format:** Use Bullet points and Bold text for clarity."
    )
    
    full_prompt = f"G5 Knowledge Base:\n{context_data}\n\nUser Inquiry: {prompt}"
    
    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": full_prompt}
        ]
        response = client.chat_completion(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=messages, 
            max_tokens=800,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- LOGGING ---
def log_to_dataset(question, answer):
    try:
        token = st.secrets["HF_TOKEN"]
        api = HfApi(token=token)
        data = {"timestamp": time.time(), "instruction": question, "output": answer}
        repo_id = "IbrahimJamal2005/construction-bot-logs" 
        api.upload_file(
            path_or_fileobj=json.dumps(data).encode("utf-8"),
            path_in_repo=f"logs/{int(time.time())}.json",
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        print(f"Log Error: {e}")

# --- DISCORD ---
def send_discord(name, phone, email, details):
    url = st.secrets.get("DISCORD_WEBHOOK_URL")
    if url:
        data = {"content": f"🏗️ **LEAD (Unified Bot):** {name}\n📞 {phone}\n📧 {email}\n📝 {details}"}
        try: requests.post(url, json=data)
        except: pass

# --- UI ---
if "is_registered" not in st.session_state:
    st.session_state.is_registered = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the G5 Construction Estimator. I can help with Roofing, Kitchens, Bathrooms, and Facade work in NYC. What project are you planning?"}]

if not st.session_state.is_registered:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏗️ G5 Construction NYC")
        st.caption("Full Service: Roofing & Interiors")
        with st.form("reg_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            desc = st.text_area("Project Details")
            if st.form_submit_button("Start Consultation"):
                if name and phone:
                    send_discord(name, phone, email, desc)
                    st.session_state.is_registered = True
                    st.rerun()
                else:
                    st.error("Name and Phone required.")
else:
    st.title("🏗️ G5 Estimator")
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Ex: Cost for a kitchen in a Co-op? OR What is the best flat roof?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        # 1. RETRIEVE DOCUMENTS (n_results=15 to catch Cross-Category info)
        # This ensures if they ask "Kitchen and Roof", the bot sees both data sets
        results = collection.query(query_texts=[prompt], n_results=15)
        context = "\n".join(results['documents'][0]) if results['documents'] else ""
        
        # Debug: Check context
        with st.sidebar:
            st.write("🔍 **Knowledge Loaded:**")
            st.caption(context[:1000] + "...")

        # 2. GENERATE RESPONSE
        with st.chat_message("assistant"):
            with st.spinner("Consulting G5 Database..."):
                resp = get_ai_response(prompt, context)
                st.write(resp)
                
        st.session_state.messages.append({"role": "assistant", "content": resp})
        log_to_dataset(prompt, resp)
