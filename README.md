💳 Smart Credit Risk Advisor
An ML & LLM-powered Financial Assistant for Personalized Credit Risk Insights

📌 Overview
This project is a multi-phase AI-powered system designed to simulate, predict, and advise users on credit risk and financial behavior. The application integrates deep learning, Markov modeling, and natural language interfaces to deliver intelligent insights based on user, card, and transaction data.

Key Highlights:

Engineered features from raw user, card, and transaction datasets.

Built a Markov model for credit score state transitions.

Developed LSTM models for spend and payment forecasting.

Simulated credit trajectories using probabilistic modeling.

Deployed an LLM-driven chatbot with Streamlit UI to answer financial and behavioral queries.

Leveraged Mistral (via Ollama) for local LLM-based assistance.

🧠 Tech Stack
Python (Pandas, NumPy, Scikit-Learn, TensorFlow/Keras)

Streamlit for UI

Ollama with Mistral 7B for natural language interaction

Matplotlib, Plotly for visualizations

📂 Folder Structure

Credit-Risk-Advisor/
│
├── Phase1_Data_Preparation.py        
├── Phase2_Feature_Engineering.py   
├── Phase3_Markov_Model.py            
├── Phase4_LSTM_Forecasting.py        
├── Phase5_Markov_Simulation.py       
├── Phase6_Credit_Assistant_Chatbot.py
├── Phase7_Visualization.py          
│
├── users_data.csv
├── cards_data.csv
├── transactions_data.csv
├── mcc_codes.json
├── train_fraud_labels.json
│
└── outputs/                          
