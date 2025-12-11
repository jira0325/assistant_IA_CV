import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama


st.title("Assistant RAG CV avec Ollama")


#  LLM local via Ollama

llm = ChatOllama(model="mistral")  


# Importation des CV

st.subheader("Chargement des CV")

uploaded_files = st.file_uploader(
    "Déposez vos CV PDF ici",
    type=["pdf"],
    accept_multiple_files=True
)

docs = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        loader = PyPDFLoader(uploaded_file)
        docs.extend(loader.load())
else:
    pdf_folder = "data/cvs"
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            docs.extend(loader.load())


# Split des textes

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)
st.write(f" {len(chunks)} chunks générés.")


# Embeddings + Chroma

emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embedding=emb)
retriever = db.as_retriever(k=3)


#  RAG 

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)


#  Analyse de l'offre d'emploi

st.subheader("Analyse de l'offre")
offer = st.text_area("Colle ici l'offre d'emploi à analyser :")

#  Analyse des CV

if st.button("Analyser les CV") and offer:

    structured_prompt = f"""
Tu es un expert en recrutement et un spécialiste de l'analyse de CV.
Analyse les CV ci-dessous en fonction de l'offre suivante :

OFFRE :
{offer}

Retourne pour chaque CV une analyse très structurée au format suivant :

###  le nom du fichier pdf
###  Score Matching (%)
Évalue la compatibilité du profil avec l'offre (0 à 100%) + justification.

###  Forces du candidat
Liste claire et structurée.

###  Gaps techniques
Toutes les compétences manquantes ou insuffisantes.

###  Recommandations
Conseils précis pour améliorer le CV ou augmenter l’adéquation au poste.

Tu dois également identifier **quel CV est le meilleur pour le poste**, avec une justification.
"""

    with st.spinner("🔍 Analyse en cours..."):
        response = qa.invoke({"query": structured_prompt})
        st.text_area("Résultat", response["result"], height=500)
