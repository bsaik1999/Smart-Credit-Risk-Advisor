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

