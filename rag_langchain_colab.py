# RAG System with LangChain, ChromaDB, and OpenAI (gpt-3.5-turbo)
# Designed for Google Colab / Jupyter Notebook use

# ============================
# 🔧 Step 1: Install Dependencies
# ============================
!pip install langchain chromadb openai tiktoken

# ============================
# 📚 Step 2: Load Environment Variables
# ============================
import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass("Enter your OpenAI API key: ")

# ============================
# 📂 Step 3: Load Documents
# ============================
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# Replace with your own documents or public corpus
docs = [
    "Customer must submit officially valid documents for identity and address proof.",
    "Aadhaar can be used as proof, but customer must consent to its use.",
    "Periodic KYC updates are mandatory for accounts older than 2 years.",
    "RBI mandates PAN for accounts above a certain threshold."
]

# Wrap as Document objects
documents = [TextLoader.from_text(text).load()[0] for text in docs]

# ============================
# ✂️ Step 4: Split into Chunks
# ============================
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# ============================
# 📊 Step 5: Embed and Store
# ============================
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embedding_model)

# ============================
# 🤖 Step 6: Build RAG Chain
# ============================
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ============================
# 💬 Step 7: Ask Questions
# ============================
while True:
    query = input("\nAsk a KYC-related question (or type 'exit'): ")
    if query.lower() == "exit":
        break
    result = rag_chain({"query": query})
    print("\nAnswer:", result["result"])
    print("\n--- Source Chunks ---")
    for doc in result["source_documents"]:
        print("•", doc.page_content)

# ============================
# 📌 To Deploy on Streamlit (Optional)
# Copy this logic into a Streamlit app using st.text_input and st.write.
# ============================
