import os
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

class RAGAgent:
    def __init__(self, index_path="faiss_index"):
        self.index_path = index_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.bm25 = None
        self.docs = []
        self.knowledge_base = [
            "Net Sales Logic: Invoices and Cancel of CN are positive values. Fresh GR, Invoice Cancel, and Expiry GR are negative values. Net Sales is the sum of these adjusted values.",
            "Pre-Ramadan Period: January 18, 2026, to February 17, 2026.",
            "Ramadan Period: February 18, 2026, to March 20, 2026.",
            "Drop Classification: Stable (<20%), Moderate (20-40%), High (40-60%), Critical (>60%).",
            "Ramadan Impact: Sales often drop due to changed consumer behavior, fasting, and logistical shifts during the holy month.",
            "Target Location: The data focus is on shops in Kolkata (Calcutta)."
        ]
        
        # Load local index if it exists
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(
                self.index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            # We still need docs for BM25, so we'll re-index if we don't have them
            # For now, we'll initialize with basic knowledge
            self._initialize_bm25(self._to_docs(self.knowledge_base))

    def _to_docs(self, texts, metadata=None):
        return [Document(page_content=t, metadata=metadata or {"type": "rule"}) for t in texts]

    def _initialize_bm25(self, docs):
        self.docs = docs
        corpus = [doc.page_content.split(" ") for doc in docs]
        self.bm25 = BM25Okapi(corpus)

    def _format_inr(self, number):
        """Formats a number into Indian Rupee format"""
        is_negative = number < 0
        number = abs(int(number))
        s = str(number)
        if len(s) <= 3:
            res = s
        else:
            res = s[-3:]
            s = s[:-3]
            while len(s) > 2:
                res = s[-2:] + "," + res
                s = s[:-2]
            res = s + "," + res
        return f"-₹{res}" if is_negative else f"₹{res}"

    def ingest_data(self, df):
        """Ingests transaction or analysis data into the hybrid index."""
        documents = []
        for i, row in df.iterrows():
            if 'Pre-Ramadan Net Sales' in df.columns:
                # We are ingesting analysis_df
                desc = (f"Shop '{row['Name']}' (ID: {row['Sold-To Party']}) had "
                        f"Pre-Ramadan sales of {self._format_inr(row['Pre-Ramadan Net Sales'])} and "
                        f"Ramadan sales of {self._format_inr(row['Ramadan Net Sales'])}, "
                        f"resulting in a {row['Drop %']:.1f}% drop. Status: {row['Classification']}.")
                content = desc
            else:
                # Fallback to basic row formatting
                content = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(
                Document(
                    page_content=content,
                    metadata={"row": i, "source": "data"}
                )
            )
        
        # Add rules as well
        rule_docs = self._to_docs(self.knowledge_base)
        all_docs = rule_docs + documents
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(all_docs)
        
        # Update Vector Store
        self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        self.vectorstore.save_local(self.index_path)
        
        # Update BM25
        self._initialize_bm25(split_docs)

    def retrieve_context(self, query, k=3):
        """Hybrid search retrieval."""
        if not self.vectorstore or not self.bm25:
            # Fallback to rules if no data ingested
            self.ingest_data(pd.DataFrame()) # Minimal init
            
        # Semantic search
        semantic_docs = self.vectorstore.similarity_search(query, k=k)
        
        # BM25 search
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        keyword_docs = [self.docs[i] for i in top_n]
        
        # Combine and remove duplicates
        combined = semantic_docs + keyword_docs
        unique_docs = list({doc.page_content: doc for doc in combined}.values())
        
        context = "\n\n".join([doc.page_content for doc in unique_docs[:k]])
        return context
