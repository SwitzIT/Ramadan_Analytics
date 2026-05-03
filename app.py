import streamlit as st
from agents.orchestrator import Orchestrator
import pandas as pd

# Page Config - Collapse sidebar
st.set_page_config(page_title="Ramadan Sales Analytics", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR PREMIUM LOOK & METRIC FIXES ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Glassmorphism KPI Cards & Text Wrapping Fix */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 15px;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        transition: transform 0.3s ease;
        white-space: normal !important; 
    }
    
    /* Force long numbers to wrap instead of truncating */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
    }

    /* Header Styling */
    .stTitle {
        font-weight: 800;
        background: linear-gradient(to right, #1e3c72, #2a5298);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        border-radius: 10px;
        padding: 0 20px;
        color: #4a5568;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2a5298 !important;
        color: white !important;
    }
    
    /* Hidden Sidebar */
    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Orchestrator
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = Orchestrator()

orchestrator = st.session_state.orchestrator

# UI Header
col1, col2 = st.columns([0.9, 0.1])
with col1:
    st.title("🌙 Ramadan Sales Performance Analytics")
    st.markdown("""
    <div style='background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #2a5298; margin-bottom: 30px;'>
        Analysis of sales trends and drops across Kolkata shops during Ramadan 2026.
    </div>
    """, unsafe_allow_html=True)

# Run Pipeline
with st.spinner("🔄 Loading data..."):
    full_df, analysis_df, chatbot_instance = orchestrator.run_pipeline()

if full_df is not None:
    # Top-right Chatbot Popover
    with col2:
        with st.popover("💬 Chat", use_container_width=True):
            st.subheader("Data Intelligence Assistant")
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            chat_container = st.container(height=350)
            with chat_container:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
            
            if prompt := st.chat_input("Ask about sales...", key="popup_chat"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        response = chatbot_instance.ask(prompt)
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # KPI Section
    st.header("📈 High-Level Metrics")
    orchestrator.viz_agent.render_kpis(analysis_df)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization Tabs
    tab1, tab2 = st.tabs(["📊 Analytics Dashboard", "📋 Detailed Table"])
    
    with tab1:
        st.markdown("<div style='background: white; padding: 20px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
        orchestrator.viz_agent.render_charts(analysis_df, full_df)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with tab2:
        classification_filter = st.multiselect(
            "Filter by Classification", 
            options=analysis_df['Classification'].unique(),
            default=analysis_df['Classification'].unique()
        )
        filtered_df = analysis_df[analysis_df['Classification'].isin(classification_filter)]
        orchestrator.viz_agent.render_table(filtered_df)
        
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Analysis",
            data=csv,
            file_name='ramadan_sales_analysis.csv',
            mime='text/csv',
        )

else:
    st.error("Failed to load data.")
