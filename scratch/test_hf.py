import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.data_engineer import DataEngineerAgent
from agents.analytics import AnalyticsAgent
from agents.rag import RAGAgent
from agents.chatbot import ChatbotAgent

def run_local_test():
    load_dotenv()
    print("--- Initializing Agents ---")
    de = DataEngineerAgent()
    analytics = AnalyticsAgent()
    rag = RAGAgent(index_path="faiss_index_test") # Use a test index
    
    print("--- Loading Data ---")
    df, error = de.load_and_clean_data()
    if error:
        print(f"Error: {error}")
        return

    print(f"Loaded {len(df)} transactions.")
    
    print("--- Analyzing Performance ---")
    analysis_df = analytics.analyze_performance(df)
    print(f"Analyzed {len(analysis_df)} shops.")

    print("--- Building/Updating Vector Database (FAISS) ---")
    rag.ingest_data(analysis_df)
    print("Vector database ready.")

    print("--- Initializing Chatbot ---")
    # This will use the repo_id="deepseek-ai/DeepSeek-V4-Pro" set in agents/chatbot.py
    chatbot = ChatbotAgent(analysis_df, rag)
    
    # Test Queries
    queries = [
        "Give me the shop whose sales drop is the most.",
        "How many shops have a critical drop?",
        "What is the average sales drop in Kolkata?"
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        response = chatbot.ask(q)
        print(f"Response:\n{response}")

if __name__ == "__main__":
    run_local_test()