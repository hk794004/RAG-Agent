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


@st.cache_resource
def get_splitters():
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)


embeddings    = get_embeddings()
text_splitters = get_splitters()
Split         = text_splitters.split_documents(all_docs)
vectorstore   = FAISS.from_documents(Split, embeddings)
retreiver     = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

with st.sidebar:
    st.info(f"🔍 Indexed {len(Split)} Chunks For Retrieval")


def _join_docs(docs, max_chars=8000):
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


qa_prompt = """
Context:
{context}

Question:
{question}

Instructions:
- Answer ONLY from the given context.
- Do not use external knowledge.
- If context contains the answer:
    Start with:
    📄 Context Founded in the provided document

    Then write:
    💡 **Answer**

    Then provide a professional, clear, slightly detailed answer.

    End with:

    📄 Source

    file_name : {file_name}
    Page_No : {page_number}

- If context does not contain the answer:
    Return ONLY:

    📄 Out of Scope --- Context Not Found in the Provided Document
"""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

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

user_input = st.chat_input("💬 Ask A Question...")

if user_input:
    st.chat_message("human").write(user_input)

    retrieved_docs  = retreiver.invoke(user_input)
    standalone_query = user_input.strip()
    context         = _join_docs(retrieved_docs)

    top_doc     = retrieved_docs[0] if retrieved_docs else None
    file_name   = top_doc.metadata.get("source_file", "Unknown") if top_doc else "Unknown"
    page_number = top_doc.metadata.get("page", "N/A") if top_doc else "N/A"

    final_prompt = qa_prompt.format(
        context=context,
        question=user_input,
        file_name=file_name,
        page_number=page_number
    )

    llm      = get_llm()
    response = llm.invoke(final_prompt)
    answer   = response.content

    history.add_user_message(user_input)
    history.add_ai_message(answer)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed = ""
        for char in answer:
            typed += char
            placeholder.markdown(typed)
            time.sleep(0.005)

    with st.expander("🔎 Debug : Rewritten Query & Retrieval"):
        st.info(f"**Query Sent to LLM : ** {standalone_query}")

    with st.expander("📄 Retrieved Chunks"):
        for i, doc in enumerate(retrieved_docs):
            src = doc.metadata.get("source_file", "Unknown")
            pg  = doc.metadata.get("page", "N/A")
            st.markdown(f"**Chunk {i+1}** | 📄 `{src}` | Page `{pg}`")
            st.text(doc.page_content[:500])
            st.divider()

with st.sidebar:
    if st.button("🧹 Clear Chat"):
        st.session_state.pop("chat_history", None)
        st.rerun()
