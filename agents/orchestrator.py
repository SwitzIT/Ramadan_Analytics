from agents.data_engineer import DataEngineerAgent
from agents.analytics import AnalyticsAgent
from agents.visualization import VisualizationAgent
from agents.rag import RAGAgent
from agents.chatbot import ChatbotAgent

class Orchestrator:
    def __init__(self):
        self.de_agent = DataEngineerAgent()
        self.analytics_agent = AnalyticsAgent()
        self.viz_agent = VisualizationAgent()
        self.rag_agent = RAGAgent()
        
    def run_pipeline(self, uploaded_file=None):
        # 1. Ingestion
        df, error = self.de_agent.load_and_clean_data(uploaded_file)
        if error:
            return None, None, error
        
        # 2. Analysis
        analysis_df = self.analytics_agent.analyze_performance(df)
        
        # 3. Update Hybrid RAG Index
        self.rag_agent.ingest_data(analysis_df)
        
        # 4. Chatbot Init
        chatbot = ChatbotAgent(analysis_df, self.rag_agent)
        
        return df, analysis_df, chatbot
