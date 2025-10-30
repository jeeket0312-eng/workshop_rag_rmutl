import streamlit as st
import os
import glob
from dotenv import load_dotenv

from langchain.schema import Document as LangChainDocument
# FIX: Import HuggingFaceEmbeddings from the correct module
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS # FIX: Import FAISS from the correct module
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

PDF_FOLDER = "./pdf"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile" # Note: llama-3.3-70b-versatile is likely a typo/non-existent model. Using llama-3.1-70b or llama-3-70b is safer. I'll keep the user's value for now.
SEARCH_LIMIT = 3

# -----------------------------
# Load PDF as text
# -----------------------------
def load_pdf_texts(folder):
    from PyPDF2 import PdfReader
    documents = []
    for pdf_file in glob.glob(f"{folder}/*.pdf"):
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            # Handle potential None from extract_text() for empty/image pages
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        documents.append(LangChainDocument(page_content=text, metadata={"source": pdf_file}))
    return documents

# -----------------------------
# RAG Chatbot
# -----------------------------
class RAGChatbot:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        # FIX: Explicitly set device to 'cpu' to prevent NotImplementedError on Streamlit Cloud
        # when it attempts to use a GPU that is unavailable/misconfigured.
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
        # NOTE: If documents is empty, FAISS.from_documents will fail. You might want to add a check here.
        if not documents:
            st.error(f"⚠️ No PDF documents found in the folder: {pdf_folder}")
            return # Exit if no documents are found
            
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

        # Conversational chain with explicit output_key for memory
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": SEARCH_LIMIT}),
            memory=self.memory,
            return_source_documents=True,
            output_key="answer"  # ต้องบอก explicitly
        )

    def answer_question(self, question, use_conversation=True):
        if use_conversation and self.conversation_chain:
            # ConversationalRetrievalChain accepts a dict with "question"
            result = self.conversation_chain({"question": question}) 
            answer = result["answer"]
            sources = result.get("source_documents", [])
        else:
            # RetrievalQA accepts a dict with "query"
            result = self.qa_chain({"query": question}) 
            answer = result["result"] # RetrievalQA uses "result" as the key
            sources = result.get("source_documents", [])
        return {"answer": answer, "sources": sources}

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Chatbot อาหารน่าน", layout="wide")
    st.title("🍽️ Chatbot อาหารจังหวัดน่าน")

    if "bot" not in st.session_state:
        # Show a loading spinner while the bot is initializing
        with st.spinner("⏳ กำลังโหลดโมเดลและเอกสาร... (อาจใช้เวลา 1-2 นาทีในการโหลดครั้งแรก)"):
            try:
                bot = RAGChatbot()
                bot.load_documents()
                st.session_state.bot = bot
            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการเริ่มต้น Chatbot: {e}")
                st.stop() # Stop the app execution if initialization fails

    bot = st.session_state.bot

    if not bot.vector_store:
        st.info("ℹ️ ไม่พบเอกสาร PDF ในโฟลเดอร์ กรุณาเพิ่มไฟล์ PDF เพื่อใช้งาน.")
        return # Do not proceed with chat if vector store is not initialized

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งข้อมูล"):
                    for s in msg["sources"]:
                        # Display source file name and a snippet of the content
                        source_path = s.metadata.get("source", "ไม่ระบุ")
                        # Only show the file name, not the full path
                        file_name = os.path.basename(source_path) 
                        st.markdown(f"**ไฟล์:** `{file_name}`")
                        st.write(s.page_content[:300] + "...") # Show slightly more context
                        st.markdown("---")


    if prompt := st.chat_input("ถามเกี่ยวกับอาหารน่าน..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display the user message
        with st.chat_message("user"):
             st.markdown(prompt)

        # Display the assistant message and response
        with st.chat_message("assistant"):
            with st.spinner("🤖 กำลังคิดคำตอบ..."):
                response = bot.answer_question(prompt, use_conversation=True)
                st.markdown(response["answer"])
                
                # Show sources immediately after the answer
                sources = response.get("sources", [])
                if sources:
                    with st.expander("📚 แหล่งข้อมูล"):
                        for s in sources:
                            source_path = s.metadata.get("source", "ไม่ระบุ")
                            file_name = os.path.basename(source_path)
                            st.markdown(f"**ไฟล์:** `{file_name}`")
                            st.write(s.page_content[:300] + "...")
                            st.markdown("---")

            # Update session state messages after displaying
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": sources
            })

if __name__ == "__main__":
    main()
