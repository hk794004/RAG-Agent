import os
import streamlit as st
import dotenv
import tempfile
import faiss
import time
import json
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="RAG Agent Portal", layout="wide")
st.title("AI Powered RAG Agent Read Multiples Documents (PDF + Word + Txt)")
st.caption("Agent Read Documents --> User Ask Question About Document --> And Get Answers")

st.divider()

with st.sidebar:
    st.header("⚙️Control")
    Api_Input = st.text_input("GROQ API KEY", type="password")

API_KEY = Api_Input if Api_Input else GROQ_API_KEY

if not API_KEY:
    raise RuntimeError("GROQ API KEY---is missing")
    st.stop()
else:
    st.sidebar.success("🔗 API Key is Running...")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload Documents",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True
    )

all_docs = []
tmp_path = []

if uploaded_file:

    for file in uploaded_file:
        extract = os.path.splitext(file.name)[1].lower()
        temp = tempfile.NamedTemporaryFile(delete=False, suffix=extract)
        temp.write(file.getvalue())
        temp.close()

        tmp_path.append((temp.name, extract, file.name))

        if extract == ".pdf":
            loader = PyPDFLoader(temp.name)
        elif extract == ".docx":
            loader = Docx2txtLoader(temp.name)
        elif extract == ".txt":
            loader = TextLoader(temp.name)
        else:
            continue

        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = file.name

        all_docs.extend(docs)

    with st.sidebar:
        st.success(f"Loaded {len(all_docs)} Pages From {len(uploaded_file)} Document")

    for clean in tmp_path:
        try:
            os.unlink(clean)
        except Exception:
            pass

if not uploaded_file:
    st.warning("📄 No Document Found")
    st.stop()

@st.cache_resource
def get_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=API_KEY,
        temperature=0.20,
        streaming=True
    )

    return llm

LLM = get_llm()

@st.cache_resource

def get_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    return embeddings

embeddings = get_embeddings()

@st.cache_resource
def get_chunks():

    text_splitters = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=100
    )

    return text_splitters

Text_Splitters = get_chunks()

Split = Text_Splitters.split_documents(all_docs)

vectorstore = FAISS.from_documents(Split, embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 20}
)

with st.sidebar:
    st.info(f"🔎 Indexed {len(Split)} Chunks For Retreived")
    
def _join_docs(docs, max_chars=7000):
    chunk, total = [], 0
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        if total + len(text) > max_chars:
            break
        source = doc.metadata.get('source_file', 'Unknown File')
        page   = doc.metadata.get('page', '?')
        header = f"[File: {source} | Page: {page}]\n"
        chunk.append(header + text)
        total += len(text)
    return "\n\n".join(chunk)


contextualize_prompt = ChatPromptTemplate.from_messages([

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
     "You are a STRICT document-only assistant.\n"
     "You ONLY answer from the provided document context.\n"
     "Zero external knowledge. Zero guessing. Zero hallucination.\n\n"

     "FIRST: Check if the answer exists in the context.\n\n"

     "If NOT found — write this one line only, then stop completely:\n"
     "📄 Out of Scope --- context not provided in the document\n\n"

     "If FOUND — write ONE response using EXACTLY this order. Each section appears once only:\n\n"

     "### 📝 Summary\n"
     "Write 3 to 5 clear lines summarizing what the document says about this topic.\n"
     "Keep it simple and easy to understand at a glance.\n\n"

     "### 💡  Answer\n"
     "Write a direct and concise answer in 2 to 4 lines from the document only.\n\n"

     "### 📖 Explanation\n"
     "Write a deep and detailed explanation using only the document context.\n"
     "Break it down step by step. Be thorough so the reader fully understands.\n"
     "Use bullet points or sub-sections where needed for clarity.\n\n"

     "### 🔑 Key Points\n"
     "Write 3 to 5 important bullet points taken directly from the document.\n\n"

     "### 📄 Source\n"
     "Write the exact file name and page number from the (File: ... | Page: ...) tags in the context.\n\n"

     "STRICT RULES:\n"
     "Write each section exactly once. Never repeat a section.\n"
     "Never add anything outside these 6 sections.\n"
     "Never use outside knowledge. Only the document context."),

    ("human",
     "## Document Context:\n{context}\n\n"
     "## Question:\n{input}")
])


if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

chat_history = st.session_state["chat_history"]

def get_history(session_id):
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

Session_ID = "default"
history = get_history(Session_ID)

def extract_suggestions(text):
    match = re.search(
        r'💡 Suggested Questions?\s*\n(.*?)(?=###|$)',
        text, re.DOTALL
    )
    if not match:
        return []
    block = match.group(1)
    return re.findall(r'-\s+(.+\?)', block)

for msg in history.messages:
    role = getattr(msg, "type", "")

    if role == "human":
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

if st.session_state.get("last_suggestions"):
    suggestions = st.session_state["last_suggestions"]

    st.markdown(
        """
        <style>
        .sugg-btn-container div[data-testid="column"] .stButton button {
            width: 100%;
            background-color: transparent;
            border: 1px solid rgba(150,150,150,0.35);
            border-radius: 12px;
            padding: 12px 16px;
            text-align: left;
            color: inherit;
            font-size: 13.5px;
            line-height: 1.5;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: normal;
        }
        .sugg-btn-container div[data-testid="column"] .stButton button:hover {
            background-color: rgba(150,150,150,0.1);
            border-color: rgba(150,150,150,0.7);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("**💡 Suggested **")

    num_cols = min(len(suggestions), 3)
    cols = st.columns(num_cols)

    st.markdown('<div class="sugg-btn-container">', unsafe_allow_html=True)
    for i, q in enumerate(suggestions):
        with cols[i % num_cols]:
            if st.button(f"🔍 {q}", key=f"sugg_{i}_{hash(q)}"):
                st.session_state["suggested_input"] = q
                st.session_state["last_suggestions"] = []
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

chat_input = st.chat_input("💬 Ask a Question...")

if "suggested_input" in st.session_state:
    User_Input = st.session_state.pop("suggested_input")
elif chat_input:
    User_Input = chat_input
else:
    User_Input = None

if User_Input:
    st.chat_message("human").write(User_Input)

    rewrite_msgs = contextualize_prompt.format_messages(
        chat_history=history.messages,
        input=User_Input,
    )

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    docs = retriever.invoke(standalone_q)

    if not docs:
        answer = "📄 Out of Scope --- context not provided in the document"
        with st.chat_message("assistant"):
            st.write(answer)
        history.add_user_message(User_Input)
        history.add_ai_message(answer)
        st.stop()

    context_str = _join_docs(docs)

    qa_msgs = qa_prompt.format_messages(
        input=User_Input,
        context=context_str,
    )

    collected = []

    def stream_once():
        for chunk in LLM.stream(qa_msgs):
            for char in chunk.content:
                collected.append(char)
                yield char
                time.sleep(0.005)

    with st.chat_message("assistant"):
        st.write_stream(stream_once())

    answer = "".join(collected)

    history.add_user_message(User_Input)
    history.add_ai_message(answer)

    sugg_prompt = f"""Based on this document answer, generate exactly 2 short follow-up questions.
    Output ONLY in this format, nothing else:
    - Question one?
    - Question two?

    Answer context:
    {answer[:1500]}"""

    sugg_response = LLM.invoke(sugg_prompt).content.strip()
    suggestions = re.findall(r'-\s+(.+\?)', sugg_response)
    st.session_state["last_suggestions"] = suggestions[:3]

    with st.expander("🔎 Debug : Rewritten Query & Retrieval"):

        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📄 Retrieved Chunks"):
        
        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p {doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

    st.rerun()

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        Chat = st.button("Clear Chat")

if Chat:
    st.session_state.pop("chat_history", None)
    st.session_state.pop("last_suggestions", None)
    st.rerun()
