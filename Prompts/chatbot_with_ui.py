import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="code.tutor()",
    page_icon="⌘",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Theme — Tokyo-Night inspired IDE palette
# ----------------------------------------------------------------------------
BG          = "#1a1b26"
BG_PANEL    = "#20222f"
BORDER      = "#2c2f42"
TEXT        = "#c8d3f5"
TEXT_DIM    = "#6b7089"
ACCENT_USER = "#e8a33d"   # amber — the ">>>" prompt
ACCENT_AI   = "#5fb3b3"   # teal  — the "#" comment
ACCENT_RED  = "#f47067"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    .stApp {{
        background-color: {BG};
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ---- Title bar, styled like an editor tab ---- */
    .titlebar {{
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 14px 18px;
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px 10px 0 0;
        margin-bottom: 0;
    }}
    .dot {{ width: 11px; height: 11px; border-radius: 50%; }}
    .dot-red {{ background: #f47067; }}
    .dot-yellow {{ background: #e8a33d; }}
    .dot-green {{ background: #7dc4a0; }}
    .titlebar-name {{
        font-family: 'JetBrains Mono', monospace;
        color: {TEXT_DIM};
        font-size: 13px;
        margin-left: 8px;
    }}

    .subhead {{
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-top: none;
        padding: 18px 20px 20px 20px;
        border-radius: 0 0 10px 10px;
        margin-bottom: 28px;
    }}
    .subhead h1 {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        color: {TEXT};
        margin: 0 0 4px 0;
        font-weight: 700;
    }}
    .subhead h1 span {{ color: {ACCENT_AI}; }}
    .subhead p {{
        color: {TEXT_DIM};
        font-size: 14px;
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* ---- Chat rows ---- */
    .row {{
        display: flex;
        margin-bottom: 22px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .prompt-tag {{
        flex-shrink: 0;
        width: 64px;
        font-weight: 700;
        font-size: 14px;
        padding-top: 2px;
    }}
    .prompt-user {{ color: {ACCENT_USER}; }}
    .prompt-ai   {{ color: {ACCENT_AI}; }}

    .bubble {{
        flex: 1;
        color: {TEXT};
        font-size: 15px;
        line-height: 1.65;
        font-family: 'Inter', sans-serif;
        padding-top: 1px;
    }}
    .bubble p {{ margin: 0 0 10px 0; }}
    .bubble p:last-child {{ margin-bottom: 0; }}
    .bubble code {{
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 4px;
        padding: 1px 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        color: {ACCENT_AI};
    }}
    .bubble pre {{
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT_AI};
        border-radius: 6px;
        padding: 12px 14px;
        overflow-x: auto;
        margin: 8px 0;
    }}
    .bubble pre code {{
        background: none;
        border: none;
        padding: 0;
        color: {TEXT};
    }}

    .timestamp {{
        color: {TEXT_DIM};
        font-size: 11px;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* ---- Empty state ---- */
    .empty {{
        text-align: center;
        padding: 60px 20px;
        color: {TEXT_DIM};
        font-family: 'JetBrains Mono', monospace;
    }}
    .empty .cursor {{
        display: inline-block;
        width: 9px;
        height: 18px;
        background: {ACCENT_AI};
        margin-left: 4px;
        animation: blink 1.1s step-end infinite;
        vertical-align: middle;
    }}
    @keyframes blink {{
        50% {{ opacity: 0; }}
    }}

    /* ---- Chat input ---- */
    [data-testid="stChatInput"] {{
        background: {BG_PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
    }}
    [data-testid="stChatInput"] textarea {{
        font-family: 'JetBrains Mono', monospace !important;
        color: {TEXT} !important;
    }}

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {{
        background: {BG_PANEL};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        font-family: 'JetBrains Mono', monospace;
        color: {TEXT};
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful assistant that teaches programming.")
    ]

if "display_log" not in st.session_state:
    st.session_state.display_log = []   # [(role, text, time_str), ...]

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### session")
    st.markdown(f"turns: `{len(st.session_state.display_log)}`")
    st.markdown("model: `gemini-3.1-flash-lite`")
    st.markdown("---")
    if st.button("clear history", use_container_width=True):
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant that teaches programming.")
        ]
        st.session_state.display_log = []
        st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown("""
<div class="titlebar">
    <div class="dot dot-red"></div>
    <div class="dot dot-yellow"></div>
    <div class="dot dot-green"></div>
    <div class="titlebar-name">tutor_session.py</div>
</div>
<div class="subhead">
    <h1>code.<span>tutor()</span></h1>
    <p># ask anything — loops, recursion, big-O, your broken code</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LLM
# ----------------------------------------------------------------------------
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

llm = get_llm()

# ----------------------------------------------------------------------------
# Render chat history
# ----------------------------------------------------------------------------
def extract_text(content):
    """Handle both plain-string and list-of-block Gemini responses."""
    if isinstance(content, list):
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return content

if not st.session_state.display_log:
    st.markdown("""
    <div class="empty">
        type a question below to start<span class="cursor"></span>
    </div>
    """, unsafe_allow_html=True)
else:
    for role, text, ts in st.session_state.display_log:
        if role == "user":
            st.markdown(f"""
            <div class="row">
                <div class="prompt-tag prompt-user">&gt;&gt;&gt;</div>
                <div>
                    <div class="bubble">{text}</div>
                    <div class="timestamp">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="row">
                <div class="prompt-tag prompt-ai"># ai</div>
                <div>
                    <div class="bubble">{text}</div>
                    <div class="timestamp">{ts}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Input
# ----------------------------------------------------------------------------
user_input = st.chat_input("ask a question...")

if user_input:
    now = datetime.now().strftime("%H:%M")

    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.display_log.append(("user", user_input, now))

    with st.spinner("thinking..."):
        result = llm.invoke(st.session_state.messages)
        ai_text = extract_text(result.content)

    st.session_state.messages.append(AIMessage(content=ai_text))
    st.session_state.display_log.append(("ai", ai_text, datetime.now().strftime("%H:%M")))

    st.rerun()