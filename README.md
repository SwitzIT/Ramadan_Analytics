# 🌙 Ramadan Sales Performance Analytics

A multi-agent Streamlit application to identify sales drops in Kolkata shops during Ramadan 2026.

## 🚀 Setup Instructions

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install Ollama (Optional for Chatbot)**:
    - Download and install Ollama from [ollama.com](https://ollama.com/).
    - Pull the Mistral model:
      ```bash
      ollama pull mistral
      ```
    - The application will automatically use a fallback logic if Ollama is not running.

3.  **Data Setup**:
    - Ensure your sales data is in the `data/` folder as `OP.xlsx`.
    - Alternatively, use the file uploader in the application sidebar.

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 🤖 Agent Architecture

- **Data Engineer Agent**: Cleans and prepares transaction data.
- **Analytics Agent**: Computes Net Sales and performance metrics.
- **Visualization Agent**: Renders interactive charts and KPI cards.
- **RAG Agent**: Manages business rules knowledge base.
- **Chatbot Agent**: Answers queries using RAG and LLM.
- **Orchestrator Agent**: Coordinates the end-to-end pipeline.

## 📊 Business Logic
- **Net Sales** = (Invoice + Cancel of CN) - (Fresh GR + Invoice Cancel + Expiry GR).
- **Comparison Period**: Jan 18 - Feb 17 (Pre-Ramadan) vs Feb 18 - Mar 20 (Ramadan).
