# --- 1. SQLITE FIX (MUST BE AT THE VERY TOP) ---
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# -----------------------------------------------

import streamlit as st
import os
import chromadb
import tempfile
import time
import json
import requests
import datetime
from chromadb.utils import embedding_functions
from huggingface_hub import InferenceClient, HfApi

# --- CONFIGURATION ---
st.set_page_config(page_title="NYC Construction Bot", page_icon="🏗️")

# --- KNOWLEDGE BASE (DATA) ---
DATA_DOCUMENTS = [
    # === ROOFING ===
    "SERVICE: Asphalt Shingles Roofing (Architectural). LOCATION: New York City (NYC). UNIT PRICE: $4.50 – $7.50 per sq ft. AVERAGE TOTAL COST (approx 1,200 sq ft): $5,400 – $9,000.",
    "SERVICE: Asphalt Shingles Roofing (Architectural). LOCATION: New York City (NYC). SCOPE & NOTES: Most common roofing for homes. Cost varies by tear-off requirements, underlayment, and ventilation needs. LIFESPAN: 20–30 years.",
    "SERVICE: Flat Roofing (TPO / EPDM). LOCATION: New York City (NYC). UNIT PRICE: $6.00 – $11.00 per sq ft. AVERAGE TOTAL COST (approx 1,200 sq ft): $7,200 – $13,200.",
    "SERVICE: Flat Roofing (TPO / EPDM). LOCATION: New York City (NYC). SCOPE & NOTES: Popular on low-slope buildings. Seams and insulation thickness affect the price. LIFESPAN: 15–30 years.",
    "SERVICE: Modified Bitumen Roofing (Torch-down). LOCATION: New York City (NYC). UNIT PRICE: $7.00 – $12.00 per sq ft. AVERAGE TOTAL COST (approx 1,200 sq ft): $8,400 – $14,400.",
    "SERVICE: Modified Bitumen Roofing (Torch-down). LOCATION: New York City (NYC). SCOPE & NOTES: Durable option for flat roofs. Requires skilled installers. LIFESPAN: 15–25 years.",
    "SERVICE: Metal Roofing (Standing Seam). LOCATION: New York City (NYC). UNIT PRICE: $10.00 – $16.00 per sq ft. AVERAGE TOTAL COST (approx 1,200 sq ft): $12,000 – $19,200.",
    "SERVICE: Metal Roofing (Standing Seam). LOCATION: New York City (NYC). SCOPE & NOTES: Higher upfront cost but has a long lifespan and low maintenance. LIFESPAN: 30–50 years.",
    "SERVICE: Roof Repair (Minor patch/flashing). LOCATION: New York City (NYC). PRICE ESTIMATE: $150 – $600 per job. NOTES: Emergency rates will increase the price.",
    "SERVICE: Roof Repair (Minor patch/flashing). LOCATION: New York City (NYC). SCOPE & NOTES: Covers small leaks and flashing work. Price depends on the extent of the repair.",

    # === STUCCO ===
    "SERVICE: 3-Coat Exterior Stucco Install (Traditional). LOCATION: New York City (NYC). UNIT PRICE: $4.00 – $8.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $4,000 – $8,000.",
    "SERVICE: 3-Coat Exterior Stucco Install (Traditional). LOCATION: New York City (NYC). SCOPE & NOTES: Includes lath, building paper, scratch coat, brown coat, and finish coat. Complexity raises cost. TIMELINE: 3–7 days.",
    "SERVICE: One-Coat Synthetic / Stucco Veneer. LOCATION: New York City (NYC). UNIT PRICE: $3.50 – $7.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $3,500 – $7,000.",
    "SERVICE: One-Coat Synthetic / Stucco Veneer. LOCATION: New York City (NYC). SCOPE & NOTES: Faster install than traditional stucco. Depends on substrate and finish type. TIMELINE: 2–5 days.",
    "SERVICE: Stucco Repair (Small cracks/patch). LOCATION: New York City (NYC). UNIT PRICE: $5.00 – $15.00 per sq ft patched. AVERAGE TOTAL COST (Typical small job): $500 – $1,500.",
    "SERVICE: Stucco Repair (Small cracks/patch). LOCATION: New York City (NYC). SCOPE & NOTES: Price varies by depth, texture match, and substrate prep. TIMELINE: 1 day.",
    "SERVICE: Full Stucco Wall Re-make (Remove and Replace). LOCATION: New York City (NYC). UNIT PRICE: $6.00 – $12.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $6,000 – $12,000.",
    "SERVICE: Full Stucco Wall Re-make (Remove and Replace). LOCATION: New York City (NYC). SCOPE & NOTES: Includes tear-off, new lath, moisture barrier, and finish coat. TIMELINE: 4–10 days.",

    # === MASONRY ===
    "SERVICE: New Full Brick Veneer (Face Brick). LOCATION: New York City (NYC). UNIT PRICE: $12.00 – $25.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $12,000 – $25,000.",
    "SERVICE: New Full Brick Veneer (Face Brick). LOCATION: New York City (NYC). SCOPE & NOTES: Includes brick ties, mortar, and scaffolding. Price varies with brick type and detailing. TIMELINE: 7–21 days.",
    "SERVICE: Full Structural Brick Wall (Solid/Backup). LOCATION: New York City (NYC). UNIT PRICE: $18.00 – $35.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $18,000 – $35,000.",
    "SERVICE: Full Structural Brick Wall (Solid/Backup). LOCATION: New York City (NYC). SCOPE & NOTES: Higher labor required. May require footings, lintels, and engineering inspections. TIMELINE: 10–28 days.",
    "SERVICE: Concrete Block (CMU) Wall (Unfinished). LOCATION: New York City (NYC). UNIT PRICE: $8.00 – $16.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $8,000 – $16,000.",
    "SERVICE: Concrete Block (CMU) Wall (Unfinished). LOCATION: New York City (NYC). SCOPE & NOTES: Typical for foundations, garages, and commercial. Grout and rebar add cost. TIMELINE: 5–14 days.",
    "SERVICE: Repointing / Tuckpointing (Brick Mortar Joints). LOCATION: New York City (NYC). UNIT PRICE: $4.50 – $12.00 per sq ft of wall. AVERAGE TOTAL COST (1,000 sq ft wall): $4,500 – $12,000.",
    "SERVICE: Repointing / Tuckpointing (Brick Mortar Joints). LOCATION: New York City (NYC). SCOPE & NOTES: Cost depends on joint depth, mortar color match, and scaffolding needs. TIMELINE: 2–7 days.",
    "SERVICE: Chimney Rebuild or Major Repair. LOCATION: New York City (NYC). PRICE ESTIMATE: $1,200 – $6,000 (Typical).",
    "SERVICE: Chimney Rebuild or Major Repair. LOCATION: New York City (NYC). SCOPE & NOTES: Includes brick crown, flue liner, flashing. Structural repairs increase price. TIMELINE: 1–7 days.",

    # === CONCRETE ===
    "SERVICE: Ready-Mix Concrete (Per Cubic Yard). LOCATION: New York City (NYC). UNIT PRICE: $181 – $218 per cubic yard. AVERAGE TOTAL COST (1,000 sq ft @ 4 inches): $223 – $269 (Material only estimate).",
    "SERVICE: Ready-Mix Concrete. LOCATION: New York City (NYC). SCOPE & NOTES: Delivery usually same day. Price varies by PSI, admixtures, and pump access.",
    "SERVICE: Concrete Slab (Pour + Finish). LOCATION: New York City (NYC). UNIT PRICE: $4.50 – $9.50 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $4,500 – $9,500.",
    "SERVICE: Concrete Slab (Pour + Finish). LOCATION: New York City (NYC). SCOPE & NOTES: Includes base prep, formwork, reinforcement. Finishes add cost. TIMELINE: 1–3 days.",
    "SERVICE: Concrete Driveway Replacement. LOCATION: New York City (NYC). UNIT PRICE: $5.50 – $12.00 per sq ft. AVERAGE TOTAL COST (1,000 sq ft): $5,500 – $12,000.",
    "SERVICE: Concrete Driveway Replacement. LOCATION: New York City (NYC). SCOPE & NOTES: Demolition, subbase, and slope/drainage affect cost. TIMELINE: 1–4 days.",
    "SERVICE: Concrete Sidewalk (New). LOCATION: New York City (NYC). UNIT PRICE: $6.00 – $11.00 per linear ft (typical 4 ft wide). AVERAGE TOTAL COST (100 ft): $600 – $1,100.",
    "SERVICE: Concrete Sidewalk (New). LOCATION: New York City (NYC). SCOPE & NOTES: Control joints and ADA ramps increase price. TIMELINE: 1 day.",
    "SERVICE: Concrete Pump Rental. LOCATION: New York City (NYC). PRICE ESTIMATE: $350 – $1,200 per day.",
    "SERVICE: Concrete Pump Rental. LOCATION: New York City (NYC). SCOPE & NOTES: Needed for hard-to-reach pours. Distance and boom height affect price.",

    # === LANDSCAPING ===
    "SERVICE: White Decorative Rock Material (Pea gravel / Crushed marble). LOCATION: New York City (NYC). UNIT PRICE: $45 – $110 per ton.",
    "SERVICE: White Decorative Rock Material. LOCATION: New York City (NYC). SCOPE & NOTES: Price varies by stone type (pea, marble chips, crushed limestone) and purity.",
    "SERVICE: White Rock Installation (Labor Only - Basic Spread). LOCATION: New York City (NYC). UNIT PRICE: $1.25 – $3.50 per sq ft. AVERAGE TOTAL COST (100 sq ft): $125 – $350.",
    "SERVICE: White Rock Installation (Labor Only). LOCATION: New York City (NYC). SCOPE & NOTES: Assumes easy access, no edging or fabric. Includes cleanup. TIMELINE: 0.5–1 day.",
    "SERVICE: Compacted Subbase (Crushed Stone). LOCATION: New York City (NYC). UNIT PRICE: $6.00 – $12.00 per sq ft (2–4 in depth). AVERAGE TOTAL COST (100 sq ft): $600 – $1,200.",
    "SERVICE: Compacted Subbase (Crushed Stone). LOCATION: New York City (NYC). SCOPE & NOTES: Needed for pathways, driveways, or heavy-use beds. TIMELINE: 1 day.",

    # === KITCHEN ===
    "SERVICE: Kitchen Design, Permit & Drawings. LOCATION: New York City (NYC). PRICE ESTIMATE: $1,200 – $8,000 (Flat fee). BREAKDOWN: Small kitchen ($1,200–$2,400), Medium ($1,800–$4,500), Large ($2,500–$8,000).",
    "SERVICE: Kitchen Design, Permit & Drawings. LOCATION: New York City (NYC). SCOPE & NOTES: Includes schematic + permit-ready drawings. Full-service design costs more. TIMELINE: 1–4 weeks.",
    "SERVICE: Kitchen Demolition & Disposal. LOCATION: New York City (NYC). PRICE ESTIMATE: $800 – $4,000 (Flat fee).",
    "SERVICE: Kitchen Demolition & Disposal. LOCATION: New York City (NYC). SCOPE & NOTES: Asbestos/lead testing or lienable building rules add cost. TIMELINE: 1–3 days.",
    "SERVICE: Kitchen Plumbing Rough-in & Fixtures. LOCATION: New York City (NYC). PRICE ESTIMATE: $1,200 – $8,000.",
    "SERVICE: Kitchen Plumbing. LOCATION: New York City (NYC). SCOPE & NOTES: Includes relocations, piping, new sink, disposal, and fixtures. TIMELINE: 1–5 days.",
    "SERVICE: Kitchen Cabinetry (Stock to Custom). LOCATION: New York City (NYC). PRICE ESTIMATE: $6,000 – $60,000. BREAKDOWN: Small kitchen ($6k–$12k), Medium ($12k–$30k), Large ($25k–$60k).",
    "SERVICE: Kitchen Cabinetry. LOCATION: New York City (NYC). SCOPE & NOTES: Stock at low end; semi-custom mid; full custom and inset cabinetry at high end. TIMELINE: 1–6 weeks (build/lead time).",
    "SERVICE: Kitchen Countertops (Laminate to Stone). LOCATION: New York City (NYC). UNIT PRICE: $25 – $200 per sq ft. PRICE ESTIMATE: $600 – $5,000 total depending on size.",
    "SERVICE: Kitchen Countertops. LOCATION: New York City (NYC). SCOPE & NOTES: Granite, quartz, and butcher block vary widely. Edge detail and undermount sinks add cost. TIMELINE: 1–3 days (template + install).",
    "SERVICE: Full Kitchen Renovation (Turnkey). LOCATION: New York City (NYC). PRICE ESTIMATE: $22,000 – $250,000+. BREAKDOWN: Small ($22k–$45k), Medium ($40k–$110k), Large/Luxury ($70k–$250k+).",
    "SERVICE: Full Kitchen Renovation (Turnkey). LOCATION: New York City (NYC). SCOPE & NOTES: Range reflects finishes, layout changes, structural or code work, and co-op approvals. TIMELINE: 4–12+ weeks.",

    # === BATHROOM ===
    "SERVICE: Bathroom Design, Permit & Drawings. LOCATION: New York City (NYC). PRICE ESTIMATE: $800 – $6,000.",
    "SERVICE: Bathroom Design, Permit & Drawings. LOCATION: New York City (NYC). TIMELINE: 1–3 weeks.",
    "SERVICE: Bathroom Plumbing Rough-in & Fixtures. LOCATION: New York City (NYC). PRICE ESTIMATE: $800 – $8,500.",
    "SERVICE: Bathroom Plumbing. LOCATION: New York City (NYC). TIMELINE: 1–7 days.",
    "SERVICE: Bathroom Tile (Walls/Floor) Material + Install. LOCATION: New York City (NYC). UNIT PRICE: $8 – $65 per sq ft. AVERAGE TOTAL COST (Medium bathroom): $800 – $3,250.",
    "SERVICE: Bathroom Tile (Walls/Floor). LOCATION: New York City (NYC). TIMELINE: 2–7 days.",
    "SERVICE: Shower / Tub Install (New). LOCATION: New York City (NYC). PRICE ESTIMATE: $900 – $12,000. BREAKDOWN: Small ($900–$1,800), Medium ($1,500–$5,000), Large/Luxury ($3,000–$12,000).",
    "SERVICE: Shower / Tub Install. LOCATION: New York City (NYC). TIMELINE: 1–7 days.",
    "SERVICE: Full Bathroom Renovation (Turnkey). LOCATION: New York City (NYC). PRICE ESTIMATE: $7,000 – $90,000+. BREAKDOWN: Small ($7k–$18k), Medium ($12k–$36k), Large ($22k–$90k+).",
    "SERVICE: Full Bathroom Renovation (Turnkey). LOCATION: New York City (NYC). TIMELINE: 2–8+ weeks.",

    # === WATERPROOFING ===
    "SERVICE: Interior Basement Crack Injection (Epoxy/Polyurethane). LOCATION: New York City (NYC). PRICE ESTIMATE: $300 – $900 per crack OR $6 – $12 per linear ft. AVERAGE TOTAL: $600 – $2,400.",
    "SERVICE: Interior Basement Crack Injection. LOCATION: New York City (NYC). SCOPE & NOTES: Best for active leaks through individual cracks. Price varies with crack length and access. TIMELINE: 1 day.",
    "SERVICE: Exterior Waterproofing Membrane (Excavate + Membrane). LOCATION: New York City (NYC). UNIT PRICE: $65 – $150 per linear ft of wall OR $25 – $60 per sq ft of wall area. AVERAGE TOTAL: $6,500 – $22,000 (typical foundation).",
    "SERVICE: Exterior Waterproofing Membrane. LOCATION: New York City (NYC). SCOPE & NOTES: Permanent solution. Includes excavation, membrane, protection board, backfill. High cost for deep excavations or confined sites. TIMELINE: 3–10 days.",
    "SERVICE: Basement Sump Pump Installation (Pump + Pit). LOCATION: New York City (NYC). PRICE ESTIMATE: $650 – $3,200.",
    "SERVICE: Basement Sump Pump Installation. LOCATION: New York City (NYC). SCOPE & NOTES: Battery backup and alarm add cost. Critical for gravity/drain systems.",

    # === DEMOLITION ===
    "SERVICE: House Demolition (Single-family, Full Demo). LOCATION: New York City (NYC). PRICE ESTIMATE: $9,500 – $26,200 (Typical NYC range).",
    "SERVICE: House Demolition. LOCATION: New York City (NYC). SCOPE & NOTES: Includes mechanical demolition, debris removal. Basements increase cost. TIMELINE: 3–14 days.",
    "SERVICE: Partial Demolition / Interior Strip-out. LOCATION: New York City (NYC). PRICE ESTIMATE: $1,500 – $8,000 per job. AVERAGE TOTAL: $2,000 – $6,000.",
    "SERVICE: Partial Demolition / Interior Strip-out. LOCATION: New York City (NYC). SCOPE & NOTES: Gutting interior finishes, MEP disconnects, selective structural removals. TIMELINE: 1–7 days.",
    "SERVICE: Concrete Slab Removal (4 inch slab). LOCATION: New York City (NYC). UNIT PRICE: $4.50 – $7.00 per sq ft. AVERAGE TOTAL (1,000 sq ft): $4,500 – $7,000.",
    "SERVICE: Concrete Slab Removal. LOCATION: New York City (NYC). SCOPE & NOTES: Costs include sawcutting, breaking, hauling. Price varies by thickness and access. TIMELINE: 1–3 days.",

    # === JUNK REMOVAL ===
    "SERVICE: Curbside Bulk Pickup (Private Hauler). LOCATION: New York City (NYC). PRICE ESTIMATE: $75 – $200 per visit.",
    "SERVICE: Curbside Bulk Pickup. LOCATION: New York City (NYC). SCOPE & NOTES: For a few items; price varies by borough and appointment window. TIMELINE: Same day–3 days.",
    "SERVICE: One-item Removal (Mattress, Sofa, Appliance). LOCATION: New York City (NYC). PRICE ESTIMATE: $50 – $200 per item.",
    "SERVICE: One-item Removal. LOCATION: New York City (NYC). SCOPE & NOTES: Heavier or bulky items cost more; includes hauling and disposal.",
    "SERVICE: Full Truckload Junk Removal. LOCATION: New York City (NYC). PRICE ESTIMATE: $500 – $1,200+ per load.",
    "SERVICE: Full Truckload Junk Removal. LOCATION: New York City (NYC). SCOPE & NOTES: Full-house cleanouts or renovation debris; disposal fees included.",
    "SERVICE: Dumpster Rental (Temporary On-site). LOCATION: New York City (NYC). PRICE ESTIMATE: $350 – $2,000+ (Rental + Haul). TYPICAL COST: $350 – $1,200.",
    "SERVICE: Dumpster Rental. LOCATION: New York City (NYC). SCOPE & NOTES: Size, permit for curb placement, and disposal tonnage affect price. TIMELINE: Rental period days–weeks.",

    # === YOUR COMPANY INFO ===
    # EDIT THESE LINES TO MATCH YOUR ACTUAL COMPANY
    "COMPANY INFO: G5 construction PHONE: (212) 555-0199. EMAIL: Info5construction.net. . INSTAGRAM: https://www.instagram.com/g5constructioncorp/.
    FACEBOOK:https://www.facebook.com/profile.php?id=61583799908215 " ,
    "COMPANY INFO: G5 construction Location. ADDRESS: 350 5th Ave, New York, NY 10118.",
    "COMPANY INFO: Business Hours. MON-FRI: 8:00 AM - 6:00 PM. SAT: 9:00 AM - 2:00 PM. SUN: Closed.",
    "COMPANY POLICY: Warranty. We offer a 5-year workmanship warranty on all structural renovations and a 1-year warranty on cosmetic finishes.",
    "COMPANY INFO: Licenses. We are fully licensed, bonded, and insured in New York City (License #1234567-DCA).",
    "COMPANY INFO: Service Areas. We serve Manhattan, Brooklyn, Queens, and parts of the Bronx."
]
DATA_IDS = [str(i) for i in range(len(DATA_DOCUMENTS))]

# --- DATABASE SETUP ---
CHROMA_DB_PATH = os.path.join(tempfile.gettempdir(), "chroma_db_data")

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="construction_knowledge", embedding_function=ef)
    
    if collection.count() == 0:
        collection.add(documents=DATA_DOCUMENTS, ids=DATA_IDS)
        
    return collection

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

# --- LOGGING FUNCTION (The Fix) ---
def log_to_dataset(question, answer):
    """
    Saves the data to your Hugging Face Dataset.
    """
    try:
        token = st.secrets["HF_TOKEN"]
        api = HfApi(token=token)
        
        data = {
            "timestamp": time.time(),
            "instruction": question,
            "output": answer
        }
        
        filename = f"logs/{int(time.time())}.json"
        
        # =========================================================
        # 👇👇👇 CHANGE THIS LINE TO YOUR USERNAME 👇👇👇
        # Example: repo_id = "jamal670/construction-bot-logs"
        repo_id = "YourUsername/construction-bot-logs" 
        # =========================================================
        
        api.upload_file(
            path_or_fileobj=json.dumps(data).encode("utf-8"),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        print("✅ Data saved to Hugging Face Dataset")
    except Exception as e:
        print(f"⚠️ Could not save log: {e}")

# --- NOTIFICATIONS ---
def send_discord(name, phone, email, details):
    url = st.secrets.get("DISCORD_WEBHOOK_URL")
    if not url: return
    data = {"content": f"🚀 **Lead:** {name} | {phone}\n{email}\n{details}"}
    try: requests.post(url, json=data)
    except: pass

# --- UI LOGIC ---
if "is_registered" not in st.session_state:
    st.session_state.is_registered = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the BuildSmart Estimator. Ask me about roofing, masonry, or kitchen costs."}]

# === PART A: REGISTRATION ===
if not st.session_state.is_registered:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2593/2593491.png", width=150)
        st.title("🏗️ Welcome")
        with st.form("reg_form"):
            name = st.text_input("Name")
            phone = st.text_input("Phone")
            email = st.text_input("Email")
            desc = st.text_area("Project")
            if st.form_submit_button("Start Chatting"):
                if name and phone:
                    send_discord(name, phone, email, desc)
                    st.session_state.is_registered = True
                    st.rerun()
                else:
                    st.error("Name and Phone required.")

# === PART B: CHAT ===
else:
    st.title("🏗️ NYC Construction Estimator")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.write(msg["content"])

    if prompt := st.chat_input("Ask about pricing..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.write(prompt)
        
        context = ""
        results = collection.query(query_texts=[prompt], n_results=3)
        if results['documents']: context = "\n".join(results['documents'][0])
        
        with st.expander("🔍 Debug: Database Context"):
            st.info(context if context else "No data found.")

        with st.chat_message("assistant"):
            with st.spinner("Estimating..."):
                resp = get_ai_response(prompt, context)
                st.write(resp)
                
        st.session_state.messages.append({"role": "assistant", "content": resp})
        
        # CALL THE LOGGING FUNCTION
        log_to_dataset(prompt, resp)
