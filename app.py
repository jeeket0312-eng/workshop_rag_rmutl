# ========================================
# app.py - Streamlit RAG Chatbot for Nan Foods
# ========================================

import streamlit as st
import os
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

# Groq LLM
from langchain_groq import ChatGroq

# PDF processing
from pypdf import PdfReader

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# Configuration
# ========================================

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    PDF_PATH = os.getenv("PDF_PATH", "./pdf/foods.pdf")
    SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "2"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    @classmethod
    def validate(cls):
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required in environment variables")
        return True

# ========================================
# PDF Loader
# ========================================

def load_pdf_to_documents(pdf_path: str) -> List[LangChainDocument]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP, separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_text(text)
    return [LangChainDocument(page_content=d, metadata={"source": Path(pdf_path).name}) for d in docs]

# ========================================
# RAG Chatbot Class
# ========================================

class RAGChatbot:
    def __init__(self):
        Config.validate()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = ChatGroq(groq_api_key=Config.GROQ_API_KEY, model_name=Config.LLM_MODEL, temperature=0.1)
        self.embeddings = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL)
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.load_documents()

    def load_documents(self):
        docs = load_pdf_to_documents(Config.PDF_PATH)
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        # QA Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": Config.SEARCH_LIMIT}),
            return_source_documents=True
        )
        # Conversational Chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": Config.SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"  
        )

    def answer_question(self, question: str, use_conversation: bool = True) -> Dict[str, Any]:
        if use_conversation and self.conversation_chain:
            history = self.memory.chat_memory.messages if self.memory else []
            result = self.conversation_chain.invoke({
                "question": question,
                "chat_history": history
            })
            answer = result.get("answer", "ไม่พบคำตอบ")
            source_docs = result.get("source_documents", [])
        else:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result", "ไม่พบคำตอบ")
            source_docs = result.get("source_documents", [])
        sources = [{"content": d.page_content[:200]+"...", "metadata": d.metadata} for d in source_docs]
        return {"answer": answer, "sources": sources}

    def clear_history(self):
        if self.memory:
            self.memory.clear()

# ========================================
# Streamlit UI
# ========================================

def main():
    st.set_page_config(page_title="Nan Foods Chatbot", page_icon="🍽️", layout="wide")
    st.title("🍽️ RAG Chatbot - อาหารจังหวัดน่าน")
    st.write("ถามข้อมูลเกี่ยวกับอาหารพื้นเมืองจังหวัดน่านได้เลย")

    # Initialize chatbot
    if "bot" not in st.session_state:
        st.session_state.bot = RAGChatbot()
    bot: RAGChatbot = st.session_state.bot

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("📚 แหล่งอ้างอิง"):
                    for s in message["sources"]:
                        st.write(f"- {s['content']}")
    
    # User input
    if prompt := st.chat_input("พิมพ์คำถามของคุณเกี่ยวกับอาหารน่าน..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                try:
                    response = bot.answer_question(prompt, use_conversation=True)
                    st.markdown(response["answer"])
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response.get("sources", [])
                    })
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {e}")

    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ ล้างประวัติการสนทนา"):
            st.session_state.chat_messages = []
            bot.clear_history()
            st.experimental_rerun()
    with col2:
        if st.button("🔄 รีเฟรชระบบ"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()
