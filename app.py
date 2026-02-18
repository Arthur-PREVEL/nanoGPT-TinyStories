import streamlit as st
import torch
import pickle
import os
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NanoGPT Storyteller",
    page_icon="✦",
    layout="centered"
)

# --- PROFESSIONAL CSS THEME ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    /* ── Reset & Base ── */
    *, *::before, *::after { box-sizing: border-box; }

    .stApp {
        background-color: #0f1117;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #161b27;
        border-right: 1px solid #1e2535;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span {
        color: #94a3b8 !important;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #e2e8f0 !important;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Slider track */
    [data-testid="stSlider"] > div > div > div {
        background-color: #4f46e5 !important;
    }

    /* ── Main title ── */
    h1 {
        font-family: 'DM Serif Display', serif !important;
        color: #f1f5f9 !important;
        font-size: 1.8rem !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em;
        margin-bottom: 0 !important;
    }

    h2, h3 {
        color: #cbd5e1 !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Subtitle / description text */
    .stApp p, .stApp span, .stApp div {
        color: #94a3b8;
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Chat container ── */
    [data-testid="stChatMessageContainer"] {
        padding: 0.25rem 0;
    }

    /* ── User bubble ── */
    [data-testid="stChatMessageUser"] {
        background-color: #1e2535 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px 12px 2px 12px !important;
        padding: 14px 18px !important;
        max-width: 80%;
        margin-left: auto;
        box-shadow: none !important;
    }

    [data-testid="stChatMessageUser"] p,
    [data-testid="stChatMessageUser"] span {
        color: #e2e8f0 !important;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* ── Assistant bubble ── */
    [data-testid="stChatMessageAssistant"] {
        background-color: #13171f !important;
        border: 1px solid #1e2535 !important;
        border-left: 3px solid #4f46e5 !important;
        border-radius: 2px 12px 12px 12px !important;
        padding: 14px 18px !important;
        box-shadow: none !important;
    }

    [data-testid="stChatMessageAssistant"] p,
    [data-testid="stChatMessageAssistant"] span {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
        line-height: 1.75;
    }

    /* Avatar */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #1e2535 !important;
        border-radius: 50%;
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"] {
        background-color: #161b27 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 12px !important;
        padding: 4px 8px !important;
    }

    [data-testid="stChatInputContainer"] textarea {
        color: #e2e8f0 !important;
        background-color: transparent !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.95rem !important;
        caret-color: #4f46e5;
    }

    [data-testid="stChatInputContainer"] textarea::placeholder {
        color: #475569 !important;
    }

    /* ── Sidebar button ── */
    .stButton > button {
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: 1px solid #2d3748 !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 8px 12px !important;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        background-color: #1e2535 !important;
        color: #e2e8f0 !important;
        border-color: #4f46e5 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-top-color: #4f46e5 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 2px; }

    /* ── Divider ── */
    hr { border-color: #1e2535 !important; }

    /* ── Error / warning ── */
    [data-testid="stAlert"] {
        background-color: #1e1520 !important;
        border-color: #7c3aed !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING LOGIC ---
REPO_ID = "Arthur-PREVEL/nanogpt-tinystories-depth24"
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model weights…"):
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            checkpoint = torch.load(path, map_location='cpu')
            config = GPTConfig(**checkpoint['model_args'])
            model = GPT(config)
            state_dict = checkpoint['model']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            model.eval()
            return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

try:
    with open('meta.pkl', 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l])
except FileNotFoundError:
    st.error("`meta.pkl` not found. Make sure it is present in the working directory.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Generation")
    temp = st.slider("Temperature", 0.1, 1.2, 0.1, step=0.05,
                     help="Higher = more creative, lower = more focused.")
    max_t = st.slider("Max tokens", 50, 500, 250, step=25,
                      help="Maximum number of characters to generate.")

    st.markdown("---")
    st.markdown("### Session")
    if st.button("↺  Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<span style='font-size:0.78rem;color:#334155'>Depth-24 Transformer · Character-level · TinyStories</span>",
        unsafe_allow_html=True
    )

# --- HEADER ---
st.title("✦ NanoGPT Storyteller")
st.markdown(
    "<p style='color:#475569;font-size:0.9rem;margin-top:-8px;margin-bottom:24px'>"
    "Give a story opening — the model will continue it.</p>",
    unsafe_allow_html=True
)

# --- CHAT HISTORY ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT ---
if prompt := st.chat_input("Once upon a time…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating…"):
            context = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            output = model.generate(context, max_t, temperature=temp)[0].tolist()
            response = decode(output)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
