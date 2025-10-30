# ========================================
# app.py - Streamlit RAG Chatbot (อาหาร)
# ========================================

import streamlit as st
import os
import glob
from dotenv import load_dotenv

from langchain.schema import Document as LangChainDocument
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -----------------------------
# Configuration
# -----------------------------
PDF_FOLDER = "./pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
SEARCH_LIMIT = 3

# -----------------------------
# PDF Loader
# -----------------------------
class DoclingLoader:
    def __init__(self):
        self.converter = DocumentConverter()
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True
        self.converter.upload_pipeline_options = {InputFormat.PDF: pdf_options}

    def load_document(self, file_path):
        result = self.converter.convert(file_path)
        doc = result.document
        text = doc.export_to_markdown()
        return LangChainDocument(page_content=text, metadata={"source": file_path})

# -----------------------------
# RAG Chatbot
# -----------------------------
class RAGChatbot:
    def __init__(self):
        self.loader = DoclingLoader()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

    def load_documents(self, pdf_folder=PDF_FOLDER):
        pdf_files = glob.glob(f"{pdf_folder}/*.pdf")
        documents = []
        for f in pdf_files:
            documents.append(self.loader.load_document(f))
        self.vector_store = FAISS.from_documents(documents, embedding=self.embeddings)

        # QA Chain
        prompt_template = """
        คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับอาหารพื้นเมืองของจังหวัดน่าน
        ข้อมูลอ้างอิง:
        {context}

        คำถาม: {question}
        คำตอบ:"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

        # Conversational Chain with explicit output_key
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"  # <-- แก้ปัญหา multiple output keys
        )

    def answer_question(self, question, use_conversation=True):
        if use_conversation and self.conversation_chain:
            result = self.conversation_chain.invoke({"question": question})
            answer = result["answer"]
            sources = result.get("source_documents", [])
        else:
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
            sources = result.get("source_documents", [])
        return {"answer": answer, "sources": sources}

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Chatbot อาหารน่าน", layout="wide")
    st.title("🍽️ Chatbot อาหารจังหวัดน่าน")
    st.markdown("ถามคำถามเกี่ยวกับอาหารพื้นเมืองของจังหวัดน่านได้เลย")

    # Initialize chatbot
    if "bot" not in st.session_state:
        st.session_state.bot = RAGChatbot()
        st.session_state.bot.load_documents()

    bot = st.session_state.bot

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งข้อมูล"):
                    for s in msg["sources"]:
                        st.write(s.metadata.get("source", "ไม่มีข้อมูล"))
                        st.write(s.page_content[:200] + "...")

    # Chat input
    if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับอาหารน่าน..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            response = bot.answer_question(prompt, use_conversation=True)
            st.markdown(response["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", [])
            })

if __name__ == "__main__":
    main()
