import streamlit as st
import dotenv
import os
import tempfile
import faiss
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="DeepDocs AI", page_icon="🤖", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600&display=swap');

/* ── Global font ── */
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   PAGE BACKGROUND  —  very soft lavender tint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
.stApp {
    background: #ffffff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SIDEBAR  —  clean white card feel
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e8e4f3 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #5b4fcf !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TITLE & CAPTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
h1 {
    background: linear-gradient(90deg, #5b4fcf, #8b6ef0) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
}
.stCaption {
    color: #8880a8 !important;
    font-size: 0.85rem !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   USER MESSAGE BUBBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ASSISTANT MESSAGE BUBBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background-color: transparent !important;
    border: none !important;
}

/* avatar */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageAvatarAssistant"] {
    background: #ede9ff !important;
    color: #5b4fcf !important;
    border-radius: 50% !important;
    border: 1.5px solid #c4b8f8 !important;
}

/* bubble */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #2d2650 !important;
    border-radius: 4px 20px 20px 20px !important;
    padding: 13px 18px !important;
    font-size: 0.91rem !important;
    line-height: 1.65 !important;
    border: 1px solid #e4deff !important;
    box-shadow: 0 2px 14px rgba(91, 79, 207, 0.08) !important;
    max-width: 82% !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
            
/* avatar */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageAvatarUser"] {
    background: #7c6ee6 !important;
    color: #ffffff !important;
    border-radius: 50% !important;
}

/* bubble */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #7c6ee6 0%, #9d8bf4 100%) !important;
    color: #ffffff !important;
    border-radius: 20px 4px 20px 20px !important;
    padding: 13px 18px !important;
    font-size: 0.91rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 4px 18px rgba(124, 110, 230, 0.22) !important;
    border: none !important;
    max-width: 78% !important;
    margin-left: auto !important;
}

/* user bubble text white rakho */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarUser"]
) [data-testid="stChatMessageContent"] *  {
    color: #ffffff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ASSISTANT MESSAGE BUBBLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageAvatarAssistant"] {
    background: #ede9ff !important;
    color: #5b4fcf !important;
    border-radius: 50% !important;
    border: 1.5px solid #c4b8f8 !important;
}

/* bubble */
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #2d2650 !important;
    border-radius: 4px 20px 20px 20px !important;
    padding: 13px 18px !important;
    font-size: 0.91rem !important;
    line-height: 1.65 !important;
    border: 1px solid #e4deff !important;
    box-shadow: 0 2px 14px rgba(91, 79, 207, 0.08) !important;
    max-width: 82% !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarAssistant"]
) [data-testid="stChatMessageContent"] * {
    color: #2d2650 !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   "human" role 
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarHuman"]
) {
    flex-direction: row-reverse !important;
}
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarHuman"]
) [data-testid="stChatMessageContent"] {
    background: linear-gradient(135deg, #7c6ee6 0%, #9d8bf4 100%) !important;
    color: #ffffff !important;
    border-radius: 20px 4px 20px 20px !important;
    padding: 13px 18px !important;
    font-size: 0.91rem !important;
    line-height: 1.65 !important;
    box-shadow: 0 4px 18px rgba(124, 110, 230, 0.22) !important;
    border: none !important;
    max-width: 78% !important;
    margin-left: auto !important;
}
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarHuman"]
) [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageAvatarHuman"]
) [data-testid="stChatMessageContent"] * {
    color: #ffffff !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   CHAT INPUT BOX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border-radius: 24px !important;
    border: 1.5px solid #d4caff !important;
    box-shadow: 0 2px 12px rgba(91, 79, 207, 0.10) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.91rem !important;
    color: #2d2650 !important;
    background: transparent !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7c6ee6 !important;
    box-shadow: 0 0 0 3px rgba(124, 110, 230, 0.15) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   BUTTONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
.stButton > button {
    background: #f0ecff !important;
    color: #5b4fcf !important;
    border: 1.5px solid #c4b8f8 !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #7c6ee6 !important;
    color: #ffffff !important;
    border-color: #7c6ee6 !important;
    box-shadow: 0 4px 14px rgba(124, 110, 230, 0.28) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   EXPANDERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e4deff !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 6px rgba(91, 79, 207, 0.06) !important;
}
[data-testid="stExpander"] summary {
    color: #5b4fcf !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ALERTS (success / info / warning)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-size: 0.86rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   FILE UPLOADER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stFileUploader"] {
    background: #faf8ff !important;
    border: 1.5px dashed #c4b8f8 !important;
    border-radius: 14px !important;
    padding: 0.5rem !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   TEXT INPUT (API key field)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1.5px solid #d4caff !important;
    font-size: 0.88rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #7c6ee6 !important;
    box-shadow: 0 0 0 3px rgba(124, 110, 230, 0.12) !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   DIVIDER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */
            
hr {
    border-color: #ede9ff !important;
    margin: 0.8rem 0 !important;
}

/* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   SUGGESTED QUESTIONS — pill buttons outside chat
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ */

.suggest-wrapper {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 4px 6px 4px;
}

.suggest-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #9b92cc;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 2px;
}

.suggest-pill {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    background: #faf8ff;
    border: 1.5px solid #ddd6ff;
    border-radius: 50px;
    padding: 9px 18px 9px 14px;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    color: #4a3fbf;
    cursor: pointer;
    transition: all 0.18s ease;
    text-decoration: none;
    width: fit-content;
    max-width: 100%;
    box-shadow: 0 1px 6px rgba(91, 79, 207, 0.07);
    white-space: normal;
    word-break: break-word;
}

.suggest-pill:hover {
    background: linear-gradient(135deg, #7c6ee6 0%, #9d8bf4 100%);
    color: #ffffff;
    border-color: transparent;
    box-shadow: 0 4px 16px rgba(124, 110, 230, 0.30);
    transform: translateY(-1px);
}

.suggest-pill .bulb {
    font-size: 1.05rem;
    flex-shrink: 0;
    filter: drop-shadow(0 1px 2px rgba(124,110,230,0.18));
}

/* Override default Streamlit button inside suggestion area */
.suggest-btn-area .stButton > button {
    background: #faf8ff !important;
    color: #4a3fbf !important;
    border: 1.5px solid #ddd6ff !important;
    border-radius: 50px !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 9px 20px !important;
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    box-shadow: 0 1px 6px rgba(91, 79, 207, 0.07) !important;
    transition: all 0.18s ease !important;
}
.suggest-btn-area .stButton > button:hover {
    background: linear-gradient(135deg, #7c6ee6 0%, #9d8bf4 100%) !important;
    color: #ffffff !important;
    border-color: transparent !important;
    box-shadow: 0 4px 16px rgba(124, 110, 230, 0.30) !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI-Powered RAG Agent — Read Multiple Documents")
st.caption("User Ask a Question About Document ➜ Agent Reads Documents ➜ Agent Generates Answers")
st.divider()

with st.sidebar:
    st.header("⚙️ Control")
    api_input = st.text_input("GROQ API KEY", type="password")

api_key = api_input if api_input else GROQ_API_KEY

if not api_key:
    raise RuntimeError("GROQ API KEY is missing")
else:
    st.sidebar.success("⚡ API KEY is Running...")

with st.sidebar:
    st.header("📁 Upload Documents")
    file_uploaded = st.file_uploader(
        "Upload Files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

all_docs = []
temp_path = []

if file_uploaded:
    for file in file_uploaded:
        ext = os.path.splitext(file.name)[1].lower()
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp.write(file.getvalue())
        temp.close()
        temp_path.append(temp.name)

        if ext == ".pdf":
            loader = PyPDFLoader(temp.name)
        elif ext == ".docx":
            loader = Docx2txtLoader(temp.name)
        elif ext == ".txt":
            loader = TextLoader(temp.name)
        else:
            continue

        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = file.name
        all_docs.extend(docs)

    with st.sidebar:
        st.success(f"Loaded {len(all_docs)} Pages From {len(file_uploaded)} Document")

if not file_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        st.warning("📃 Documents Not Found!")
        st.stop()

    for clean in temp_path:
        try:
            os.unlink(clean)
        except Exception:
            pass


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

embeddings    = get_embeddings()

@st.cache_resource
def get_splitters():

    splitters = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120
    )

    return splitters

text_splitters = get_splitters()

Split = text_splitters.split_documents(all_docs)

vectorstore   = FAISS.from_documents(
    Split,
    embeddings
)

retreiver     = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

with st.sidebar:
    st.info(f"🔍 Indexed {len(Split)} Chunks For Retrieval")

def _join_docs(docs, max_chars=7000):
    chunk, total = [], 0
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            break
        chunk.append(text)
        total += len(text)
    return "\n\n".join(chunk)

Context_Prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are a query reformulation engine for a strict document-only RAG system.\n"
     "Rewrite the user's question into a clean standalone search query using chat history.\n"
     "Resolve pronouns, merge follow-ups, remove filler words.\n"
     "Output ONLY the reformulated query. Nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")

])

qa_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are a Strict RAG Assistant You Only Ansawer The Provided Document:\n\n"

     "═══════════════════════════════════════════════\n"
     "MODE 1 — CASUAL CHAT (greetings, small talk, off-topic)\n"
     "═══════════════════════════════════════════════\n"
     "If the user says hi, hello, salam, how are you, thanks, jokes,\n"
     "or anything NOT related to document research:\n"
     "→ Reply warmly and naturally like a friendly assistant.\n"
     "→ Be witty, kind, and conversational.\n"
     "→ At the end, gently mention you can help with their documents.\n"
     "→ No headers. No sections. Just plain friendly text.\n\n"

     "═══════════════════════════════════════════════\n"
     "MODE 2 — DOCUMENT Q&A (questions about uploaded documents)\n"
     "═══════════════════════════════════════════════\n"
     "Supported formats: PDF, TXT, DOCX\n"
     "Zero external knowledge. Zero guessing. Zero hallucination.\n\n"

     "FIRST: Check if the answer exists in the context.\n\n"

     "If NOT found — reply casually like:\n"
     "'Hmm, I couldn't find that in your document. Maybe try rephrasing? 🤔'\n\n"

     "If FOUND — write the response in exactly this format:\n\n"

     "[Your clean flowing paragraph answer here. No headings. No icons. No bullet points.\n"
     "Just clear, natural, well-written paragraph(s) from the document context only.\n"
     "Be thorough but readable.]\n\n"

     "---\n"
     "📄 Source\n"
     "File : [exact file name from context]\n"
     "Page : [exact page number from context]\n\n"

    "STRICT RULES:\n"
    "NEVER use ###, icons, or section titles inside the answer.\n"
    "The separator --- and Source block must always appear at the very end.\n"
    "File and Page must each be on their own separate line.\n"
    "STRICTLY answer from context only.\n"
    "Do not infer.\m"
    "Do not suggest document structure.\n"
    "Never use outside knowledge. Only the document context."),

    ("human",
     "## Document Context:\n{context}\n\n"
     "## Question:\n{input}")
])

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

if "suggested_questions" not in st.session_state:
    st.session_state["suggested_questions"] = []

if "pending_question" not in st.session_state:
    st.session_state["pending_question"] = None

chat_history = st.session_state["chat_history"]

def get_history(session_id):
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

Session_ID = "default"
history    = get_history(Session_ID)

for message in history.messages:
    role = getattr(message, "type", "")
    if role == "human":
        st.chat_message("human").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)

def generate_suggestions(retrieved_docs, standalone_query):
    if not retrieved_docs:
        return [
            "What are the main topics in this document?",
            "Can you summarize this document?"
        ]

    text = " ".join([d.page_content for d in retrieved_docs[:2]])[:2000]
    llm  = get_llm()

    suggestion_prompt = f"""
You are a smart assistant that generates 2 short follow-up questions.

Based ONLY on the document context below and user query,
generate 2 helpful and relevant questions the user might ask next.

RULES:
- Only output 2 questions
- No explanations
- No numbering text
- Keep it simple and clear

USER QUERY:
{standalone_query}

DOCUMENT CONTEXT:
{text}

OUTPUT FORMAT:
1. ...
2. ...
"""
    res     = llm.invoke(suggestion_prompt).content.strip().split("\n")
    cleaned = []
    for r in res:
        r = r.strip("- ").strip()
        if r:
            cleaned.append(r)

    return cleaned[:2] if len(cleaned) >= 2 else [
        "Can you explain this topic in detail?",
        "What are the key points in this section?"
    ]

suggestions = st.session_state["suggested_questions"]

if suggestions:
    st.markdown("***💡 Suggested Questions***", unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="suggest-btn-area">', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            q1_text = suggestions[0] if len(suggestions) > 0 else ""
            if q1_text:
                if st.button(f" {q1_text}", key="sq_0", use_container_width=True):
                    st.session_state["pending_question"] = q1_text
                    st.rerun()

        with col_b:
            q2_text = suggestions[1] if len(suggestions) > 1 else ""
            if q2_text:
                if st.button(f" {q2_text}", key="sq_1", use_container_width=True):
                    st.session_state["pending_question"] = q2_text
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

pending    = st.session_state.pop("pending_question", None)
user_input = st.chat_input("💬 Ask A Question...") or pending

if user_input:
    st.chat_message("human").write(user_input)

    past_messages = history.messages  

    llm = get_llm()

    context_chain = Context_Prompt | llm
    rewritten      = context_chain.invoke({
        "chat_history": past_messages,
        "input": user_input
    })
    standalone_query = rewritten.content.strip().strip('"')

    retrieved_docs = retreiver.invoke(standalone_query)

    def build_context(docs):
        parts = []
        for doc in docs:
            file_name = doc.metadata.get("source_file", "Unknown")
            file_type = doc.metadata.get("type", "UNKNOWN")      
            page_num  = doc.metadata.get("page", "N/A")
            parts.append(
                f"(File: {file_name} | Type: {file_type} | Page: {page_num})\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(parts)

    context = build_context(retrieved_docs)

    qa_chain = qa_prompt | llm
    response = qa_chain.invoke({
        "context": context,
        "input":   standalone_query
    })
    answer = response.content

    history.add_user_message(user_input)
    history.add_ai_message(answer)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed = ""
        for char in answer:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.005)

    new_suggestions = generate_suggestions(retrieved_docs, standalone_query)
    st.session_state["suggested_questions"] = new_suggestions[:2]

    with st.expander("🔎 Debug : Rewritten Query & Retrieval"):
        st.info(f"**Original Query:** {user_input}")
        st.success(f"**Rewritten Standalone Query:** {standalone_query}")

    with st.expander("📄 Retrieved Chunks"):
        for i, doc in enumerate(retrieved_docs):
            src  = doc.metadata.get("source_file", "Unknown")
            pg   = doc.metadata.get("page", "N/A")
            ftype = doc.metadata.get("type", "UNKNOWN")
            st.markdown(f"**Chunk {i+1}** | 📄 `{src}` | Type `{ftype}` | Page `{pg}`")
            st.text(doc.page_content[:500])
            st.divider()

    st.rerun()

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state["suggested_questions"] = []
        st.session_state["pending_question"]    = None
        st.rerun()
