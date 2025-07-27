import streamlit as st
import pandas as pd
import plotly.express as px
import requests

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

    st.header(f"📋 Credit Summary for User ID: `{user_id}`")

    st.markdown("### 🔑 Key Metrics")
    st.markdown(f"- **💳 Credit Limit:** `${user_df['LIMIT_BAL'].iloc[0]:,.0f}`")
    st.markdown(f"- **🎂 Age:** `{user_df['AGE'].iloc[0]}` years")
    st.markdown(f"- **📉 Avg. Bill Amount:** `${user_df['avg_bill_amt'].iloc[0]:,.2f}`")
    st.markdown(f"- **📈 Avg. Payment Amount:** `${user_df['avg_pay_amt'].iloc[0]:,.2f}`")
    st.markdown(f"- **📊 Credit Utilization Ratio:** `{user_df['credit_util_ratio'].iloc[0]:.2f}`")
    st.markdown(f"- **💵 Payment Ratio:** `{user_df['payment_ratio'].iloc[0]:.2f}`")
    st.markdown(f"- **📌 Mean Payment Status:** `{user_df['mean_pay_status'].iloc[0]:.2f}`")
    st.markdown(f"- **⚠️ Max Payment Status:** `{user_df['max_pay_status'].iloc[0]}`")

    st.divider()
    st.markdown("### 📊 Distribution Comparison")

    fig1 = px.histogram(df, x="avg_bill_amt", nbins=50, title="Avg Bill Amount Distribution",
                        color_discrete_sequence=["#3B82F6"])
    fig1.add_vline(x=user_df['avg_bill_amt'].iloc[0], line_dash="dash", line_color="red")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="credit_util_ratio", nbins=50, title="Credit Utilization Ratio Distribution",
                        color_discrete_sequence=["#10B981"])
    fig2.add_vline(x=user_df['credit_util_ratio'].iloc[0], line_dash="dash", line_color="red")
    st.plotly_chart(fig2, use_container_width=True)

# --- Main Streamlit App ---
def main():
    st.set_page_config(page_title="Smart Credit Risk Advisor", layout="wide")

    st.title("📊 Smart Credit Risk Advisor with Ollama")

    data = load_user_data()

    user_list = data['ID'].unique()
    selected_user = st.selectbox("Select User ID to View Credit Summary", user_list)

    if selected_user:
        show_user_summary(data, selected_user)

    # --- Sidebar chat assistant with full chat history ---
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 💬 Chat Assistant")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history as chat bubbles
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(
                    f"<div style='padding:10px; background:#e1f5fe; border-radius:10px; margin-bottom:8px;'>"
                    f"<strong>You:</strong><br>{chat['user']}</div>", unsafe_allow_html=True
                )
            with st.chat_message("assistant"):
                st.markdown(
                    f"<div style='padding:10px; background:#fce4ec; border-radius:10px; margin-bottom:8px;'>"
                    f"<strong>Assistant:</strong><br>{chat['assistant']}</div>", unsafe_allow_html=True
                )

        # Input box for new user message
        user_input = st.text_input("Ask your Credit Assistant 👇", key="chat_input")

        if user_input:
            # Show user message immediately
            with st.chat_message("user"):
                st.markdown(
                    f"<div style='padding:10px; background:#e1f5fe; border-radius:10px; margin-bottom:8px;'>"
                    f"<strong>You:</strong><br>{user_input}</div>", unsafe_allow_html=True
                )

            # Prepare context including user data for prompt
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
                f"Question: {user_input}"
            )

            with st.spinner("Thinking..."):
                reply = ask_ollama(context)

            # Show assistant response
            with st.chat_message("assistant"):
                st.markdown(
                    f"<div style='padding:10px; background:#fce4ec; border-radius:10px; margin-bottom:8px;'>"
                    f"<strong>Assistant:</strong><br>{reply}</div>", unsafe_allow_html=True
                )

            # Append to chat history
            st.session_state.chat_history.append({"user": user_input, "assistant": reply})

            # Clear input box by rerunning app
            st.experimental_rerun()


if __name__ == "__main__":
    main()
