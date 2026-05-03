import os
import pandas as pd
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from rank_bm25 import BM25Okapi

load_dotenv()

def load_excel(file_path):
   df = pd.read_excel(file_path)
   documents = []
   for i, row in df.iterrows():
       content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
       documents.append(
           Document(
               page_content=content,
               metadata={"row": i}
           )
       )
   return documents

# =========================
# 2. SPLIT
# =========================
def split_docs(documents):
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50
   )
   return splitter.split_documents(documents)

# =========================
# 3. EMBEDDINGS + VECTOR DB
# =========================
def create_vectorstore(docs):
   embeddings = HuggingFaceEmbeddings(
       model_name="sentence-transformers/all-MiniLM-L6-v2"
   )
   if os.path.exists("faiss_index"):
       db = FAISS.load_local(
           "faiss_index",
           embeddings,
           allow_dangerous_deserialization=True
       )
   else:
       db = FAISS.from_documents(docs, embeddings)
       db.save_local("faiss_index")
   return db, embeddings

# =========================
# 4. BM25 (KEYWORD SEARCH)
# =========================
def create_bm25(docs):
   corpus = [doc.page_content.split(" ") for doc in docs]
   bm25 = BM25Okapi(corpus)
   return bm25, docs

# =========================
# 5. HYBRID RETRIEVER
# =========================
def hybrid_search(query, vectorstore, bm25, docs, k=3):
   # Semantic search
   semantic_docs = vectorstore.similarity_search(query, k=k)
   # BM25 search
   tokenized_query = query.split(" ")
   bm25_scores = bm25.get_scores(tokenized_query)
   top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
   keyword_docs = [docs[i] for i in top_n]
   # Combine
   combined = semantic_docs + keyword_docs
   # Remove duplicates
   unique_docs = list({doc.page_content: doc for doc in combined}.values())
   return unique_docs[:k]

# =========================
# 6. LLM
# =========================
def load_llm():
   llm = HuggingFaceEndpoint(
       repo_id="meta-llama/Llama-3.1-8B-Instruct",
       task="text-generation",
       temperature=0.3,
       max_new_tokens=512
   )
   return ChatHuggingFace(llm=llm)

# =========================
# 7. FINAL QA FUNCTION
# =========================
def answer_query(query, vectorstore, bm25, docs, llm):
   retrieved_docs = hybrid_search(query, vectorstore, bm25, docs)
   context = "\n\n".join([doc.page_content for doc in retrieved_docs])
   prompt = f"""
You are a helpful AI assistant.
Answer ONLY from the provided context.
If answer is not found, say "Not found in data".
Context:
{context}
Question:
{query}
Answer:
"""
   response = llm.invoke(prompt)
   return {
       "answer": response.content,
       "sources": [doc.metadata for doc in retrieved_docs]
   }
