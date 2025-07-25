# 💳 Smart Credit Risk Advisor

🧠 A high-level ML + LLM-powered engine for analyzing, forecasting, and advising on credit risk using user transactions, card behavior, and demographics.

## 🚀 What It Does

- Cleans and prepares real-world banking datasets (`users`, `cards`, `transactions`)
- Engineers behavioral features like:
  - Online transaction ratio
  - Refund frequency
  - Monthly average spend
  - Credit utilization trends
- Models **credit score transitions using Markov Chains**
- Forecasts user repayment behavior via **LSTM time series**
- Provides:
  - 🤖 A **Streamlit chatbot assistant** powered by Mistral + Ollama
  - 📉 A **Markov simulation engine** for future credit state predictions
  - 📊 Interactive plots and KPI visualizations

## 🧠 Core Tech Stack

- Python (Pandas, NumPy, Scikit-Learn)
- LSTM (TensorFlow/Keras)
- Markov Chains (custom transition matrix modeling)
- Streamlit (chatbot + visual insights)
- Ollama + Mistral 7B (LLM integration)
- Plotly / Matplotlib (data visualization)

## 📁 File Structure

```bash
├── Phase1_Data_Preparation.py         # Cleans and merges users, cards, and transactions
├── Phase2_Feature_Engineering.py      # Extracts credit behavior KPIs
├── Phase3_Markov_Model.py             # Constructs credit state transitions
├── Phase4_LSTM_Forecasting.py         # Predicts monthly spend via LSTM
├── Phase5_Markov_Simulation.py        # Simulates credit behavior using Markov paths
├── Phase6_Credit_Assistant_Chatbot.py # Streamlit chatbot assistant (LLM-based)
├── Phase7_Visualization.py            # Trend plots and user-level dashboards
│
├── users_data.csv
├── cards_data.csv
├── transactions_data.csv
├── mcc_codes.json
├── train_fraud_labels.json
│
├── outputs/                           # Folder for saving plots, results, simulations
├── requirements.txt
├── README.md
