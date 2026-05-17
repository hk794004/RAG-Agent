import streamlit as st
import os
import tempfile
import dotenv
import json
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

st.set_page_config(page_title="RAG Agent Portal", layout="wide")
st.title("🤖 AI Powered RAG Agent Read Multiples Document (PDF + Word + TXT)")
st.caption("Agent Read Documents---->Ask Question----> Get Answers")
st.divider()

with st.sidebar:

    st.header("⚙️Control")

    api_input = st.text_input(
        "GROQ_API_KEY",
        type="password"
    )

api_key = api_input if api_input else GROQ_API_KEY

if not api_key:
    st.error("🔑GROQ API KEY Missing")
    st.stop()
else:
    st.sidebar.success("🔗 API Connected & Running")

with st.sidebar:
    
    pdf_uploader = st.file_uploader(
        "Upload Document",
        type=["pdf","docx","txt"],
        accept_multiple_files=True
    )

if not pdf_uploader:
    st.info("Upload PDFs into the Sidebar")
    st.stop()

all_docs = []
temp_path = []

if pdf_uploader:

    for pdf in pdf_uploader:
        ext = os.path.splitext(pdf.name)[1].lower()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp.write(pdf.getvalue())
        temp.close()

        temp_path.append((temp.name, ext, pdf.name))

        if ext==".pdf":
            loader = PyPDFLoader(temp.name)
        elif ext==".docx":
            loader = Docx2txtLoader(temp.name)
        elif ext==".txt":
            loader = TextLoader(temp.name)

        docs = loader.load()

        for d in docs:
            d.metadata["source_file"] = pdf.name

        all_docs.extend(docs)

    st.sidebar.success(f"Loaded {len(all_docs)} Pages From {len(pdf_uploader)} Document")

    for clean in temp_path:
        try:
            os.unlink(clean)
        except Exception as e:
            pass

@st.cache_resource
def get_embediings():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings" : True}
    )

    return embeddings

embeddings = get_embediings()

@st.cache_resource
def get_llm():

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=api_key,
        temperature=0,
        streaming=False
    )

    return llm

@st.cache_resource
def chunks():

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=120
    )

    return text_splitter

Text_Splitters = chunks()

Split = Text_Splitters.split_documents(all_docs)

vectorstore = FAISS.from_documents(
    Split,
    embeddings
)

retreiver = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k" : 5, "fetch_k" : 20}
)

st.sidebar.info(f"🔍 indexed {len(Split)} Chunks for Retreived")

def _join_docs(docs, max_chars=7000):
    chunk, total = [], 0
                                                    
    for d in docs:
        piece = d.page_content
        if total + len(piece) > max_chars:
            break
        chunk.append(piece)
        total += len(piece)
    return "\n\n---\n\n".join(chunk)

contextualize_q_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are an expert assistant that rewrites user questions into standalone questions for document retrieval.\n"
     "Your job is to use the chat history ONLY if needed to understand context.\n\n"
     
     "Rules:\n"
     "- Convert the user's question into a clear, self-contained question\n"
     "- Do NOT answer the question\n"
     "- Do NOT add explanations\n"
     "- Do NOT assume missing facts not in chat history\n"
     "- Keep it short and precise\n"
     "- The output must be only the rewritten question"
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")

])

qa_prompt = ChatPromptTemplate.from_messages([

    ("system",
     "You are an expert AI assistant specialized in analyzing documents.\n\n"

     "Your role:\n"
     "- Answer ONLY using the given context from the document\n"
     "- Do NOT use outside knowledge\n"
     "- If the answer is NOT in the context, do NOT say 'not available'\n"
     "  Instead: Suggest 2-3 related topics or stories that ARE present in the document\n"
     "  Format:\n"
     "  'This topic is not in the document, but here are related things I found:\n"
     "   • [suggestion 1]\n"
     "   • [suggestion 2]'\n\n"

     "- Never hallucinate\n"
     "- Always stay within document content only"
    ),

    ("human",
     "Context from document:\n{context}\n\n"
     "Question:\n{input}")

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

for msg in history.messages:
    role = getattr(msg, "type", "")

    if role=="human":
        st.chat_message("human").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)

User_Input = st.chat_input("💬 Ask a Question...")

if User_Input:
    st.chat_message("human").write(User_Input)

    rewrite_msgs = contextualize_q_prompt.format_messages(
        chat_history=history.messages, 
        input=User_Input,
    )

    LLM = get_llm()

    standalone_q = LLM.invoke(rewrite_msgs).content.strip()

    docs = retreiver.invoke(standalone_q)

    if not docs:
        answer = "Out of Scope -- Context is not found in the provided documents."
        
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

    answer = LLM.invoke(qa_msgs).content

    with st.chat_message("assistant"):
        st.markdown(answer)

    history.add_user_message(User_Input)
    history.add_ai_message(answer)

    with st.expander("🔍 Debug : Rewritten Query & Retrieval"):

        st.write("**Rewritten (standalone) query:**")
        st.code(standalone_q or "(empty)", language="text")
        st.write(f"**Retrieved {len(docs)} chunk(s).**")

    with st.expander("📄 Retrieved Chunks"):

        for i, doc in enumerate(docs, 1):
            st.markdown(f"**{i}. {doc.metadata.get('source_file','Unknown')} (p {doc.metadata.get('page','?')})**")
            st.write(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

with st.sidebar:

    col1,col2 = st.columns(2)

    with col1:
        Chat = st.button("Clear Chat")

if Chat:
    st.session_state.pop("chat_history", None)
    st.rerun()

export_data = []

for message in history.messages:
    role = getattr(message, "type", "")

    if role=="human":
        export_data.append({"role" : "human", "message" : message.content})
    else:
        export_data.append({"role" : "assistant", "message" : message.content})


json_data = json.dumps(export_data, ensure_ascii=False, indent=2)

with st.sidebar:

    with col2:

        download_button = st.download_button(
            "Chat_History",
            data=json_data,
            file_name="chat_history.json"
        )







   
