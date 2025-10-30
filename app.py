import streamlit as st
import os
import glob
from dotenv import load_dotenv

from langchain.schema import Document as LangChainDocument
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
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
LLM_MODEL = "llama-3.3-70b-versatile"
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
            text += page.extract_text() + "\n"
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
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vector_store = None
        self.qa_chain = None
        self.conversation_chain = None
        self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=LLM_MODEL, temperature=0.1)

    def load_documents(self, pdf_folder=PDF_FOLDER):
        documents = load_pdf_texts(pdf_folder)
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
        bot = RAGChatbot()
        bot.load_documents()
        st.session_state.bot = bot

    bot = st.session_state.bot

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 แหล่งข้อมูล"):
                    for s in msg["sources"]:
                        st.write(s.metadata.get("source", "ไม่ระบุ"))
                        st.write(s.page_content[:200] + "...")

    if prompt := st.chat_input("ถามเกี่ยวกับอาหารน่าน..."):
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
