import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json

# --- Helper to call Ollama LLM via HTTP ---
def ask_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11435/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.RequestException as e:
        return f"Error communicating with Ollama: {e}"

# --- Load user credit data ---
@st.cache_data
def load_user_data():
    df = pd.read_csv(r"C:\Users\SaiKrishna\OneDrive\Desktop\archive (3)\credit_data_processed.csv")
    return df

# --- Display user credit summary and plots ---
def show_user_summary(df, user_id):
    user_df = df[df['ID'] == user_id]

    st.header(f"Credit Summary for User ID: {user_id}")

    st.write("### Key Metrics")
    st.write(f"**Credit Limit:** ${user_df['LIMIT_BAL'].iloc[0]:,.0f}")
    st.write(f"**Age:** {user_df['AGE'].iloc[0]}")
    st.write(f"**Average Bill Amount:** ${user_df['avg_bill_amt'].iloc[0]:,.2f}")
    st.write(f"**Average Payment Amount:** ${user_df['avg_pay_amt'].iloc[0]:,.2f}")
    st.write(f"**Credit Utilization Ratio:** {user_df['credit_util_ratio'].iloc[0]:.2f}")
    st.write(f"**Payment Ratio:** {user_df['payment_ratio'].iloc[0]:.2f}")
    st.write(f"**Mean Payment Status:** {user_df['mean_pay_status'].iloc[0]:.2f}")
    st.write(f"**Max Payment Status:** {user_df['max_pay_status'].iloc[0]}")

    st.write("---")

    st.write("### Distribution Comparison")
    fig1 = px.histogram(df, x="avg_bill_amt", nbins=50, title="Distribution of Average Bill Amount")
    fig1.add_vline(x=user_df['avg_bill_amt'].iloc[0], line_dash="dash", line_color="red")
    st.plotly_chart(fig1)

    fig2 = px.histogram(df, x="credit_util_ratio", nbins=50, title="Distribution of Credit Utilization Ratio")
    fig2.add_vline(x=user_df['credit_util_ratio'].iloc[0], line_dash="dash", line_color="red")
    st.plotly_chart(fig2)

# --- Main Streamlit App ---
def main():
    st.title("📊 Smart Credit Risk Advisor with Ollama")

    data = load_user_data()

    user_list = data['ID'].unique()
    selected_user = st.selectbox("Select User ID to View Credit Summary", user_list)

    if selected_user:
        show_user_summary(data, selected_user)

    st.write("---")
    st.header("🤖 Ask your Credit Assistant")

    user_question = st.text_area("Ask any question about your credit, loans, or payments:")

    if user_question:
        user_df = data[data['ID'] == selected_user].iloc[0]
        context = (
            f"User Data:\n"
            f"Credit Limit: {user_df['LIMIT_BAL']}\n"
            f"Age: {user_df['AGE']}\n"
            f"Average Bill Amount: {user_df['avg_bill_amt']:.2f}\n"
            f"Average Payment Amount: {user_df['avg_pay_amt']:.2f}\n"
            f"Credit Utilization Ratio: {user_df['credit_util_ratio']:.2f}\n"
            f"Payment Ratio: {user_df['payment_ratio']:.2f}\n"
            f"Mean Payment Status: {user_df['mean_pay_status']:.2f}\n"
            f"Max Payment Status: {user_df['max_pay_status']}\n\n"
            f"Question: {user_question}"
        )

        with st.spinner("Thinking..."):
            answer = ask_ollama(context)
            st.success(f"**Assistant:** {answer}")

if __name__ == "__main__":
    main()
