import streamlit as st
import torch
import pickle
from huggingface_hub import hf_hub_download
from model import GPTConfig, GPT

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NanoGPT Storyteller",
    page_icon="📖",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #f7f8fa; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #111827 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'Inter', sans-serif !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #6b7280 !important;
    font-size: 0.85rem !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] hr { border-color: #f3f4f6 !important; margin: 1.25rem 0 !important; }

[data-testid="stSlider"] > div > div > div { background-color: #4f46e5 !important; }

/* ── Clear button ── */
.stButton > button {
    background-color: #ffffff !important;
    color: #374151 !important;
    border: 1px solid #d1d5db !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 8px 14px !important;
}
.stButton > button:hover {
    background-color: #f3f4f6 !important;
    border-color: #9ca3af !important;
}

/* ── Main area ── */
.block-container { padding-top: 2.5rem !important; max-width: 720px !important; }

h1 {
    font-family: 'Lora', serif !important;
    color: #111827 !important;
    font-size: 1.65rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0 !important;
}

/* ── Chat row ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0.2rem 0 !important;
    align-items: flex-start !important;
    gap: 0.65rem !important;
}

/* ── User bubble ── */
[data-testid="stChatMessageUser"] {
    background-color: #4f46e5 !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 11px 16px !important;
    border: none !important;
    box-shadow: 0 1px 4px rgba(79,70,229,0.2) !important;
}
[data-testid="stChatMessageUser"] p,
[data-testid="stChatMessageUser"] span {
    color: #ffffff !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Assistant bubble ── */
[data-testid="stChatMessageAssistant"] {
    background-color: #ffffff !important;
    border-radius: 4px 18px 18px 18px !important;
    padding: 11px 16px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatMessageAssistant"] p,
[data-testid="stChatMessageAssistant"] span {
    color: #1f2937 !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Chat input ── */
[data-testid="stChatInputContainer"] {
    background-color: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 14px !important;
    padding: 2px 6px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatInputContainer"] textarea {
    color: #111827 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.93rem !important;
    background: transparent !important;
    caret-color: #4f46e5;
}
[data-testid="stChatInputContainer"] textarea::placeholder { color: #9ca3af !important; }
[data-testid="stChatInputContainer"] button {
    background-color: #4f46e5 !important;
    border-radius: 8px !important;
    border: none !important;
}

.stSpinner > div { border-top-color: #4f46e5 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 2px; }
[data-testid="stAlert"] { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
REPO_ID  = "Arthur-PREVEL/nanogpt-tinystories-depth24"
FILENAME = "ckpt.pt"

@st.cache_resource
def load_model():
    try:
        with st.spinner("Loading model…"):
            path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
            checkpoint = torch.load(path, map_location="cpu")
            config = GPTConfig(**checkpoint["model_args"])
            model = GPT(config)
            state_dict = {
                k.replace("_orig_mod.", ""): v
                for k, v in checkpoint["model"].items()
            }
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
    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join([itos[i] for i in l])
except FileNotFoundError:
    st.error("`meta.pkl` not found. Make sure it is present in the working directory.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    temp  = st.slider("Temperature", 0.1, 1.2, 0.1, step=0.05,
                      help="Higher = more creative. Lower = more focused.")
    max_t = st.slider("Max tokens", 50, 500, 250, step=25,
                      help="Maximum number of characters to generate.")
    st.markdown("---")
    if st.button("↺  Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem;color:#9ca3af;line-height:1.6'>"
        "Depth-24 Transformer<br>Character-level · TinyStories</p>",
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("NanoGPT Storyteller")
st.markdown(
    "<p style='color:#6b7280;font-size:0.88rem;margin-top:-4px;margin-bottom:28px'>"
    "Give a story opening — the model will continue it.</p>",
    unsafe_allow_html=True,
)

# ── Chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# Passing emoji as avatar= replaces the Material icon entirely — no overlap
for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "📖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Once upon a time…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Generating…"):
            context  = torch.tensor(encode(prompt), dtype=torch.long)[None, ...]
            output   = model.generate(context, max_t, temperature=temp)[0].tolist()
            response = decode(output)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
