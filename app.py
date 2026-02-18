import streamlit as st
import torch
import pickle
import time
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CSS (THEME FIX)
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="UiT nanoGPT Assistant", 
    page_icon="✨", 
    layout="centered"
)

st.markdown("""
    <style>
    /* 1. FORCE TEXT COLOR TO DARK GLOBALLY (Fixes the white-on-white issue) */
    .stApp, .stMarkdown, .stText, p, h1, h2, h3, li, span {
        color: #0f172a !important; /* Dark Slate Blue (Almost Black) */
        font-family: 'Source Sans Pro', sans-serif;
    }

    /* 2. BACKGROUNDS */
    .stApp { background-color: #ffffff; } /* White Background */
    [data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }

    /* 3. CHAT BUBBLES */
    
    /* User Bubble (Green) */
    [data-testid="stChatMessageUser"] { 
        background-color: #059669; 
        border: none;
    }
    /* Force User Text to White */
    [data-testid="stChatMessageUser"] p, [data-testid="stChatMessageUser"] span { 
        color: #ffffff !important; 
    }

    /* Assistant Bubble (Light Gray) */
    [data-testid="stChatMessageAssistant"] { 
        background-color: #f1f5f9; 
        border: 1px solid #e2e8f0;
    }

    /* 4. DIALOG / MODAL FIX */
    div[data-testid="stDialog"] {
        background-color: #ffffff;
    }
    div[data-testid="stDialog"] p, div[data-testid="stDialog"] h3 {
        color: #0f172a !important;
    }

    /* 5. UI ELEMENTS */
    .stButton>button { border-radius: 8px; font-weight: 500; }
    div[data-baseweb="select"] > div { border-radius: 8px; }
    
    /* Hide default header decoration */
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. CONSTANTS
# -----------------------------------------------------------------------------

REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24" 
FILENAME = "ckpt.pt"

SUGGESTIONS = {
    "Magic": "Tim found a magic ball in the garden. He picked it up and...",
    "Forest": "Lily was walking through the big green forest when she saw a...",
    "Robot": "In a small house, there lived a robot named Ben. Ben wanted to...",
    "Rain": "It was a rainy day. Sue was sad because she could not go outside...",
}

# -----------------------------------------------------------------------------
# 3. BACKEND (MODEL LOADING)
# -----------------------------------------------------------------------------

@st.cache_resource
def load_engine():
    try:
        # Load Model
        path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        checkpoint = torch.load(path, map_location='cpu')
        config = GPTConfig(**checkpoint['model_args'])
        model = GPT(config)
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
        model.load_state_dict(state_dict)
        model.eval()

        # Load Meta (Tokenizer)
        with open('meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        return model, meta['stoi'], meta['itos']
    except Exception as e:
        return None, None, None

model, stoi, itos = load_engine()

if not model:
    st.error("❌ Error: Could not load model. Check internet connection or 'meta.pkl'.")
    st.stop()

encode = lambda s: [stoi[c] for c in s if c in stoi]
decode = lambda l: ''.join([itos[i] for i in l])

def generate_text(prompt, max_tokens, temp):
    context = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
    out = model.generate(context, max_new_tokens=max_tokens, temperature=temp)[0].tolist()
    return decode(out)

# -----------------------------------------------------------------------------
# 4. SIDEBAR (CONTROLS & INFO)
# -----------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    
    # Sliders for Control
    max_tokens = st.slider("Story Length (Tokens)", 50, 500, 250, step=10)
    temperature = st.slider("Creativity (Temperature)", 0.1, 1.2, 0.8, step=0.1)
    
    st.divider()
    
    # Project Info Button (Opens Dialog)
    @st.dialog("📋 Project Information")
    def show_info():
        st.markdown("""
        ### Scaling Laws in Small Language Models
        
        **Model Architecture:**
        * **Type:** Transformer (Decoder-only)
        * **Depth:** 24 Layers
        * **Parameters:** ~18.95 Million
        * **Dataset:** TinyStories
        
        **Context:**
        This model demonstrates that **depth** (number of layers) is the primary driver for narrative logic and consistency in small models, outperforming wider but shallower architectures.
        
        *Author: Arthur PREVEL - UiT*
        """)
    
    if st.button("ℹ️ About this Model", use_container_width=True):
        show_info()
        
    if st.button("🗑️ Clear Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# -----------------------------------------------------------------------------
# 5. MAIN INTERFACE
# -----------------------------------------------------------------------------

# Init History
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- HERO SECTION (If Empty) ---
if not st.session_state.messages:
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: #059669 !important;'>UiT nanoGPT</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b !important; font-size: 1.2rem;'>A 24-layer Transformer trained on TinyStories.</p>", unsafe_allow_html=True)
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

# --- CHAT DISPLAY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT HANDLING ---
user_input = st.chat_input("Write the start of a story...")

# Pills Logic (Only show if empty)
selected_pill = None
if not st.session_state.messages:
    # Using columns to center the pills visually
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        selected_key = st.pills("Try a starter:", options=SUGGESTIONS.keys(), selection_mode="single")
        if selected_key:
            selected_pill = SUGGESTIONS[selected_key]

# --- EXECUTION LOGIC ---
final_prompt = user_input if user_input else selected_pill

if final_prompt:
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # 2. Assistant Generation
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_text = ""
        
        with st.spinner("Dreaming up a story..."):
            # Run Inference
            generated_text = generate_text(final_prompt, max_tokens, temperature)
            
            # Simple Streaming Effect
            for word in generated_text.split():
                full_text += word + " "
                placeholder.markdown(full_text + "▌")
                time.sleep(0.03) 
                
            placeholder.markdown(full_text)
    
    st.session_state.messages.append({"role": "assistant", "content": full_text})
    
    # Rerun to remove pills if they were just clicked
    if selected_pill:
        st.rerun()
