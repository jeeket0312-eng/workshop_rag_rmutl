import streamlit as st
import os
import glob
from dotenv import load_dotenv

# Import standard LangChain components
from langchain.schema import Document as LangChainDocument
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Import components from the community packages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# Import PyPDF2 for PDF loading
try:
    from PyPDF2 import PdfReader
except ImportError:
    st.error("❌ ต้องติดตั้ง PyPDF2: pip install PyPDF2")
    st.stop()


# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Configuration Constants ---
PDF_FOLDER = "./pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
SEARCH_LIMIT = 2      # Reduce to 2 to help with token size
MAX_DOC_LENGTH = 1500 # Limit context per document

# -----------------------------
# Load PDF as text
# -----------------------------
def load_pdf_texts(folder):
    """Loads text from all PDF files in the specified folder and truncates content."""
    documents = []
    # Check if the folder exists
    if not os.path.exists(folder):
        st.error(f"⚠️ โฟลเดอร์เอกสารไม่พบ: {folder}")
        return documents

    for pdf_file in glob.glob(f"{folder}/*.pdf"):
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            # Truncate text if too long
            text = text[:MAX_DOC_LENGTH]
            if text.strip():
                documents.append(LangChainDocument(page_content=text, metadata={"source": pdf_file}))
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถอ่านไฟล์ PDF: {pdf_file} (ข้อผิดพลาด: {e})")
            continue
            
    return documents

# -----------------------------
# RAG Chatbot Class
# -----------------------------
class RAGChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            input_key="question"
        )
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'} # ใช้ CPU เพื่อความเสถียร
        )
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        
        # ตรวจสอบ API Key ก่อนใช้งาน
        if not GROQ_API_KEY:
            st.error("❌ กรุณาตั้งค่า GROQ_API_KEY ในไฟล์ .env")
            st.stop()
            
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

    def load_documents(self, pdf_folder=PDF_FOLDER):
        documents = load_pdf_texts(pdf_folder)
        if not documents:
            # st.error ถูกเรียกใน load_pdf_texts แล้วหาก folder ไม่พบ
            if os.path.exists(pdf_folder) and not glob.glob(f"{pdf_folder}/*.pdf"):
                 st.error(f"⚠️ ไม่พบเอกสาร PDF ในโฟลเดอร์: {pdf_folder}")
            return False
            
        self.vector_store = FAISS.from_documents(documents, embedding=self.embeddings)

        # --- 1. RetrievalQA Chain Setup ---
        prompt_template = """
คุณเป็นผู้ช่วยเรื่องอาหารพื้นเมืองจังหวัดน่าน
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

        # --- 2. ConversationalRetrievalChain Setup (แก้ไขข้อผิดพลาด) ---
        # ลบ chain_type_kwargs={"output_key": "answer"} ออก
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True,
        )
        return True

    def answer_question(self, question, use_conversation=True):
        """Answers the user's question using the selected chain."""
        if use_conversation and self.conversation_chain:
            # ใช้ invoke สำหรับ ConversationalRetrievalChain
            result = self.conversation_chain.invoke({"question": question}) 
            answer = result.get("answer", "ไม่สามารถหาคำตอบได้") # ใช้ .get() เพื่อป้องกัน KeyError
            sources = result.get("source_documents", [])
        else:
            # ใช้ invoke สำหรับ RetrievalQA Chain
            result = self.qa_chain.invoke({"query": question}) 
            answer = result.get("result", "ไม่สามารถหาคำตอบได้")
            sources = result.get("source_documents", [])
        return {"answer": answer, "sources": sources}

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Chatbot อาหารน่าน", layout="wide")
    st.title("🍽️ Chatbot อาหารจังหวัดน่าน")
    st.markdown("---")

    if "bot" not in st.session_state:
        # Initialize and load documents only once
        st.session_state.bot_error = False
        with st.spinner("⏳ กำลังโหลดโมเดลและเอกสาร..."):
            try:
                bot = RAGChatbot()
                if bot.load_documents():
                    st.session_state.bot = bot
                else:
                    # ถ้าโหลดเอกสารล้มเหลว แต่ไม่มีข้อผิดพลาดร้ายแรง (เช่นแค่ไม่มีไฟล์)
                    st.session_state.bot_error = True 
            except Exception as e:
                # ข้อผิดพลาดร้ายแรงในการเริ่มต้น เช่น API Key หรือโมเดล
                st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น Chatbot: {e}")
                st.stop()
    
    # ถ้ามีการแจ้ง error ในการโหลดเอกสาร (แต่ bot ยังถูกสร้างอยู่)
    if "bot_error" in st.session_state and st.session_state.bot_error:
        # ป้องกันไม่ให้โค้ดส่วนอื่นทำงานหากไม่มี vector_store
        if "bot" in st.session_state and not st.session_state.bot.vector_store:
             st.info("ℹ️ ไม่พบเอกสาร PDF ในโฟลเดอร์ กรุณาเพิ่มไฟล์ PDF ใน `./pdf` เพื่อใช้งาน.")
        return

    bot = st.session_state.bot
    
    # ------------------ Chat History Display ------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Keep only last 5 messages to reduce token usage
    if len(st.session_state.messages) > 10: # เพิ่มเป็น 10 เพื่อให้มีบริบทมากขึ้น
        st.session_state.messages = st.session_state.messages[-10:]

    # Display history messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                    for s in msg["sources"]:
                        source_path = s.metadata.get("source", "ไม่ระบุ")
                        file_name = os.path.basename(source_path) 
                        st.markdown(f"**ไฟล์:** `{file_name}`")
                        st.write(s.page_content[:300] + "...") 
                        st.markdown("---")

    # ------------------ Handle new user input ------------------
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารน่าน..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 กำลังคิดคำตอบ..."):
                try:
                    # ใช้ conversational chain เป็นหลัก
                    response = bot.answer_question(prompt, use_conversation=True) 
                    st.markdown(response["answer"])
                    
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                            for s in sources:
                                source_path = s.metadata.get("source", "ไม่ระบุ")
                                file_name = os.path.basename(source_path)
                                st.markdown(f"**ไฟล์:** `{file_name}`")
                                st.write(s.page_content[:300] + "...")
                                st.markdown("---")
                                
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาดขณะตอบคำถาม: {e}")
                    st.session_state.messages.pop() # ลบคำถามล่าสุดออกหากเกิดข้อผิดพลาด

if __name__ == "__main__":
    main()
