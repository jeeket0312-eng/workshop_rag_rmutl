# app.py - RAG Chatbot สำหรับอาหารจังหวัดน่าน
import streamlit as st
import logging
import os
import glob
from pathlib import Path
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document as LangChainDocument
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import glob

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== CONFIG =====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_SIZE = 384
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
SEARCH_LIMIT = 3
PDF_PATH = "./pdf/foods.pdf"
SYSTEM_MESSAGE = "คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับอาหารพื้นเมืองของจังหวัดน่าน ตอบคำถามอย่างกระชับและถูกต้อง"

# ===================== PDF Loader =====================
class DoclingDocumentLoader:
    def __init__(self):
        self.converter = DocumentConverter()
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = True
        pdf_options.do_table_structure = True
        pdf_options.table_structure_options.do_cell_matching = True
        self.converter.upload_pipeline_options = {InputFormat.PDF: pdf_options}

    def load_document(self, file_path: str) -> LangChainDocument:
        result = self.converter.convert(file_path)
        doc = result.document
        text_content = doc.export_to_markdown()
        langchain_doc = LangChainDocument(
            page_content=text_content,
            metadata={"source": file_path, "title": doc.name, "file_type": Path(file_path).suffix}
        )
        return langchain_doc

# ===================== RAG Service =====================
class FoodRAGChatbot:
    def __init__(self):
        self.doc_loader = DoclingDocumentLoader()
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder="./model_cache")
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

    def create_vector_store(self, document: LangChainDocument):
        # Split document
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        docs = splitter.split_documents([document])
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def setup_qa_chain(self):
        prompt_template = """
        คุณเป็นผู้ช่วยที่เชี่ยวชาญเกี่ยวกับอาหารพื้นเมืองของจังหวัดน่าน ตอบคำถามอย่างกระชับและถูกต้อง

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

    def setup_conversation_chain(self):
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True
        )

    def load_pdf(self, pdf_path=PDF_PATH):
        doc = self.doc_loader.load_document(pdf_path)
        self.create_vector_store(doc)
        self.setup_qa_chain()
        self.setup_conversation_chain()
        
     def answer_question(self, question: str, use_conversation=False):
        if use_conversation and self.conversation_chain:
            result = self.conversation_chain.invoke({"question": question})
            # บางเวอร์ชันใช้ key 'answer', บางเวอร์ชันใช้ 'result'
            answer = result.get("answer") or result.get("result") or "ไม่พบคำตอบจากโมเดล"
            source_docs = result.get("source_documents", [])
        else:
            result = self.qa_chain.invoke({"query": question})
            answer = result.get("result") or result.get("answer") or "ไม่พบคำตอบจากโมเดล"
            source_docs = result.get("source_documents", [])

# ===================== Streamlit UI =====================
def main():
    st.set_page_config(page_title="Chatbot อาหารจังหวัดน่าน", page_icon="🍽️", layout="wide")
    st.title("🍽️ RAG Chatbot - อาหารพื้นเมืองจังหวัดน่าน")
    st.markdown("---")

    # Initialize chatbot
    if "rag_bot" not in st.session_state:
        bot = FoodRAGChatbot()
        bot.load_pdf()
        st.session_state.rag_bot = bot
        st.session_state.messages = []

    bot = st.session_state.rag_bot

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งอ้างอิง"):
                    for i, src in enumerate(msg["sources"],1):
                        st.write(f"{i}. {src['content']}")
                        if src.get("metadata"):
                            st.write(f"ข้อมูลเพิ่มเติม: {src['metadata']}")

    # Chat input
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารพื้นเมืองของจังหวัดน่าน..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                response = bot.answer_question(prompt, use_conversation=True)
                st.markdown(response["answer"])
                if response.get("sources"):
                    with st.expander("📚 แหล่งอ้างอิง"):
                        for i, src in enumerate(response["sources"],1):
                            st.write(f"{i}. {src['content']}")
                            if src.get("metadata"):
                                st.write(f"ข้อมูลเพิ่มเติม: {src['metadata']}")
                # Save response
                st.session_state.messages.append({"role":"assistant","content":response["answer"],"sources":response.get("sources")})

if __name__ == "__main__":
    main()
