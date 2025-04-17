import sqlite3
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# --- App Setup ---
st.set_page_config(page_title="LogWise 👁️‍🗨️", layout="wide")
st.title("🤖 LogWise — AI-Powered Log Chatbot")
st.markdown("Chat with your logs! Upload a file and interact with AI for anomaly detection and insights — all locally!")

DB_NAME = "logs.db"

# --- Database Utilities ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            level TEXT,
            message TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            message TEXT
        )
    ''')
    conn.commit()
    conn.close()

def store_logs(df):
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("logs", conn, if_exists="append", index=False)
    conn.close()

def fetch_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df

def detect_anomalies(df):
    if df.empty or len(df) < 10:
        return df, []
    df['length'] = df['message'].str.len()
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['length']])
    anomalies = df[df['anomaly'] == -1]
    return df, anomalies

def parse_log(file):
    lines = file.read().decode('utf-8').splitlines()
    data = []
    for line in lines:
        if ' - ' in line:
            try:
                parts = line.split(' - ', 2)
                timestamp, level, message = parts
                data.append({'timestamp': timestamp, 'level': level, 'message': message})
            except:
                continue
    return pd.DataFrame(data)

def store_chat(role, message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (role, message) VALUES (?, ?)", (role, message))
    conn.commit()
    conn.close()

def fetch_chat():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM chat_history", conn)
    conn.close()
    return df

# --- Initialize ---
init_db()

# --- Sidebar Chat History ---
with st.sidebar:
    st.header("🗨️ Chat History")
    chat_df = fetch_chat()
    if not chat_df.empty:
        for _, row in chat_df.iterrows():
            st.markdown(f"**{row['role']}**: {row['message']}")

# --- Main Chat Area ---
st.subheader("💬 Chat Interface")

# Upload Section (chat style)
st.chat_message("user").markdown("📂 Upload your log file (`.log` or `.txt`) to begin.")

uploaded_file = st.file_uploader("Choose a log file", type=["log", "txt"])
if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        message = f"✅ {len(log_df)} log entries added successfully!"
        st.chat_message("assistant").success(message)
        store_chat("assistant", message)
    else:
        message = "⚠️ No valid log lines found. Please upload a well-formatted `.log` file."
        st.chat_message("assistant").error(message)
        store_chat("assistant", message)

# Show Logs
logs_df = fetch_logs()
if not logs_df.empty:
    st.chat_message("user").markdown("📋 Show me the logs.")
    st.chat_message("assistant").dataframe(logs_df[['timestamp', 'level', 'message']], use_container_width=True)

    # Anomaly detection
    st.chat_message("user").markdown("🔍 Can you detect any anomalies?")
    analyzed_df, anomalies_df = detect_anomalies(logs_df)
    if not anomalies_df.empty:
        st.chat_message("assistant").markdown("🚨 **Anomalies found!** Here's what looks suspicious:")
        st.chat_message("assistant").dataframe(anomalies_df[['timestamp', 'level', 'message']], use_container_width=True)
    else:
        st.chat_message("assistant").markdown("✨ All clear! No anomalies detected.")

    # Chart
    st.chat_message("assistant").markdown("📈 Here's a quick look at the message length distribution:")
    fig, ax = plt.subplots()
    ax.hist(analyzed_df['length'], bins=20, color='salmon', edgecolor='black')
    ax.set_xlabel("Message Length")
    ax.set_ylabel("Frequency")
    st.chat_message("assistant").pyplot(fig)
else:
    st.info("Upload a log file to start chatting with LogWise!")
