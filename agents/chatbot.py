import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import pandas as pd
import streamlit as st

load_dotenv()

class ChatbotAgent:
    def __init__(self, analysis_df, rag_agent):
        self.df = analysis_df
        self.rag = rag_agent
        self.llm = self._load_llm()

    def _load_llm(self):
        try:
            # Priority 1: Streamlit Secrets (for Deployment)
            # Priority 2: OS Environment Variables (for Local .env)
            token = None
            if "HF_TOKEN" in st.secrets:
                token = st.secrets["HF_TOKEN"]
            elif "HUGGINGFACEHUB_API_TOKEN" in st.secrets:
                token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
            
            if not token:
                token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

            if not token:
                print("Warning: HF_TOKEN not found in secrets or .env")
                return None
            
            # Using Llama 3.1 model per user request
            endpoint = HuggingFaceEndpoint(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                task="text-generation",
                temperature=0.3,
                max_new_tokens=512,
                huggingfacehub_api_token=token
            )
            return ChatHuggingFace(llm=endpoint)
        except Exception as e:
            print(f"Error loading HuggingFace LLM: {e}")
            return None

    def ask(self, query):
        context = self.rag.retrieve_context(query, k=5)
        
        # Calculate Aggregate Stats for Management Summary
        avg_drop = self.df['Drop %'].mean()
        cat_counts = self.df['Classification'].value_counts()
        
        # Identify Outliers
        top_drops = self.df.sort_values(by='Drop %', ascending=False).head(5)
        top_performers = self.df.sort_values(by='Drop %', ascending=True).head(5)
        
        summary_context = f"""
MANAGEMENT SUMMARY STATISTICS:
- Total Stores Analyzed: {len(self.df)}
- Overall Average Sales Drop: {avg_drop:.2f}%
- Store Classification Counts:
  * Stable (<20% drop): {cat_counts.get('Stable', 0)}
  * Moderate Drop (20-40%): {cat_counts.get('Moderate Drop', 0)}
  * High Drop (40-60%): {cat_counts.get('High Drop', 0)}
  * Critical Drop (>60%): {cat_counts.get('Critical Drop', 0)}

ABSOLUTE OUTLIERS (TOP DROPS):
"""
        for _, r in top_drops.iterrows():
            summary_context += f"- {r['Name']}: {r['Drop %']:.2f}% drop ({r['Classification']})\n"
            
        summary_context += "\nTOP PERFORMERS (LEAST DROP):\n"
        for _, r in top_performers.iterrows():
            summary_context += f"- {r['Name']}: {r['Drop %']:.2f}% drop ({r['Classification']})\n"

        if self.llm:
            prompt = f"""
You are an expert executive business analyst assistant for a Kolkata sales dashboard.
You must answer questions accurately and professionally.
Use the provided Context and Dataset Summary to answer the User Query.

Retrieved Context (Semantic matches):
{context}

Dataset Summary (Aggregate Stats & Outliers):
{summary_context}

User Query:
{query}

Answer:
"""
            try:
                response = self.llm.invoke(prompt)
                return response.content
            except Exception as e:
                print(f"LLM Inference Error: {e}")
                return self._fallback_lookup(query)
        else:
            return self._fallback_lookup(query)

    def _fallback_lookup(self, query):
        return "I'm sorry, I am operating in fallback mode because the HuggingFace API key (HF_TOKEN) is missing or the endpoint is unavailable. Please check your .env file or terminal logs."
