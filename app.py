from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM — Groq (free)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

# Embeddings — local, completely free
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load document
def load_documents():
    with open("documents/fmcg.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text

# Strategy 1 — Large chunks, less overlap
def strategy1_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Strategy 2 — Small chunks, more overlap
def strategy2_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk) for chunk in chunks]

# Build vector store
def build_vectorstore(docs, collection_name):
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./chroma_db"
    )

# RAG chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template("""
You are an expert in FMCG supply chains and AI systems.
Answer the question based ONLY on the following context.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")

    def rag_chain(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "context": context,
            "question": question
        })
        return response, docs

    return rag_chain

# Main
if __name__ == "__main__":
    print("Loading documents...")
    text = load_documents()
    print(f"✅ Document loaded — {len(text)} characters")

    print("\nBuilding vector stores...")
    docs1 = strategy1_chunks(text)
    docs2 = strategy2_chunks(text)
    print(f"✅ Strategy 1: {len(docs1)} chunks")
    print(f"✅ Strategy 2: {len(docs2)} chunks")

    vs1 = build_vectorstore(docs1, "strategy1")
    vs2 = build_vectorstore(docs2, "strategy2")
    print("✅ Vector stores built")

    chain1 = build_rag_chain(vs1)
    chain2 = build_rag_chain(vs2)

    # Quick test
    question = "What is demand forecasting?"
    print(f"\n🔍 Test Question: {question}")

    answer1, _ = chain1(question)
    print(f"\n📦 Strategy 1 Answer:\n{answer1}")

    answer2, _ = chain2(question)
    print(f"\n📦 Strategy 2 Answer:\n{answer2}")