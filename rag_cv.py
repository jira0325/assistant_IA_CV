from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import sys, os

sys.stdout.reconfigure(encoding='utf-8')

# LLM local (pas Groq)
llm = ChatOllama(model="mistral")   

# 1. Load PDFs
docs = []
pdf_folder = "data/cvs"

for file in os.listdir(pdf_folder):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder, file))
        docs.extend(loader.load())

print(f"{len(docs)} pages PDF chargées.")

# 2. Split
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks générés.")

# 3. Embeddings
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(chunks, embedding=emb)
retriever = db.as_retriever(k=3)

# 4. RAG
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# 5. Question
question = "Quel CV correspond le mieux à un Data Scientist NLP ?"
response = qa.invoke({"query": question})

print(response["result"])
