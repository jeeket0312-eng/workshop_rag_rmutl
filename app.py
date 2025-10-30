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
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

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
    from PyPDF2 import PdfReader
    documents = []
    for pdf_file in glob.glob(f"{folder}/*.pdf"):
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
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

    def load_documents(self, pdf_folder=PDF_FOLDER):
        documents = load_pdf_texts(pdf_folder)
        if not documents:
            st.error(f"⚠️ ไม่พบเอกสาร PDF ในโฟลเดอร์: {pdf_folder}")
            return False
            
        self.vector_store = FAISS.from_documents(documents, embedding=self.embeddings)

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

        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True
            # chain_type_kwargs={"output_key": "answer"}  # <-- REMOVE THIS LINE
        )
        return True

    def answer_question(self, question, use_conversation=True):
        """Answers the user's question using the selected chain."""
        if use_conversation and self.conversation_chain:
            result = self.conversation_chain({"question": question}) 
            answer = result["answer"]
            sources = result.get("source_documents", [])
        else:
            result = self.qa_chain({"query": question}) 
            answer = result["result"]
            sources = result.get("source_documents", [])
        return {"answer": answer, "sources": sources}

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Chatbot อาหารน่าน", layout="wide")
    st.title("🍽️ Chatbot อาหารจังหวัดน่าน")

    if "bot" not in st.session_state:
        # Initialize and load documents only once
        with st.spinner("⏳ กำลังโหลดโมเดลและเอกสาร..."):
            try:
                bot = RAGChatbot()
                if bot.load_documents():
                    st.session_state.bot = bot
                else:
                    st.session_state.bot_error = True # Flag if document loading failed
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น Chatbot: {e}")
                st.stop()
    
    if "bot_error" in st.session_state and st.session_state.bot_error:
        return

    bot = st.session_state.bot
    
    if not bot.vector_store:
        st.info("ℹ️ ไม่พบเอกสาร PDF ในโฟลเดอร์ กรุณาเพิ่มไฟล์ PDF เพื่อใช้งาน.")
        return 

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Keep only last 5 messages to reduce token usage
    if len(st.session_state.messages) > 5:
        st.session_state.messages = st.session_state.messages[-5:]

    # Display history messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งข้อมูล"):
                    for s in msg["sources"]:
                        source_path = s.metadata.get("source", "ไม่ระบุ")
                        file_name = os.path.basename(source_path) 
                        st.markdown(f"**ไฟล์:** `{file_name}`")
                        st.write(s.page_content[:300] + "...") 
                        st.markdown("---")

    # Handle new user input
    if prompt := st.chat_input("ถามเกี่ยวกับอาหารน่าน..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 กำลังคิดคำตอบ..."):
                try:
                    response = bot.answer_question(prompt, use_conversation=True)
                    st.markdown(response["answer"])
                    
                    sources = response.get("sources", [])
                    if sources:
                        with st.expander("📚 แหล่งข้อมูล"):
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
                    st.session_state.messages.pop() 

if __name__ == "__main__":
    main()
