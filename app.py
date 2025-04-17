import sqlite3
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# --- App Setup ---
st.set_page_config(page_title="LogWise 👁️‍🗨️", layout="wide")
st.title("👁️‍🗨️ LogWise — AI-Powered Local Log Analyzer")
st.markdown("Welcome to **LogWise**! Upload your log file and watch it work: 🧠 **pattern analysis**, 🚨 **anomaly detection**, and 🗄️ **local storage** — no cloud involved!")

DB_NAME = "logs.db"

# --- Reset Database Every Refresh ---
def reset_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS logs")
    cursor.execute("DROP TABLE IF EXISTS chat_history")
    conn.commit()
    conn.close()

# --- Initialize Tables ---
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
        try:
            if ' - ' in line:
                parts = line.split(' - ', 2)
                timestamp = parts[0]
                level = parts[1]
                message = parts[2]
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

# --- Run Initialization ---
reset_db()   # <--- Clears database every refresh
init_db()    # <--- Then re-creates tables

# --- Upload Log File ---
st.chat_message("user").markdown("📂 Upload a log file (.log or .txt) to get started:")
uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])

if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        message = "✅ Logs successfully stored in the local database."
        st.chat_message("assistant").success(message)
        store_chat("assistant", message)
    else:
        message = "⚠️ No valid log entries found in the file."
        st.chat_message("assistant").error(message)
        store_chat("assistant", message)

# --- Display Chat History in Sidebar with Manage Options ---
with st.sidebar:
    st.subheader("🗨️ Chat History (Persistent)")
    chat_df = fetch_chat()
    if not chat_df.empty:
        selected_index = st.selectbox("Select message to manage:", chat_df.index, format_func=lambda i: f"{chat_df.at[i, 'role'].capitalize()}: {chat_df.at[i, 'message'][:40]}...")

        if selected_index is not None:
            selected_msg = chat_df.loc[selected_index]

            # Rename option
            new_message = st.text_input("✏️ Rename this message:", value=selected_msg['message'])
            if st.button("Update Message"):
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("UPDATE chat_history SET message = ? WHERE id = ?", (new_message, selected_msg['id']))
                conn.commit()
                conn.close()
                st.experimental_rerun()

            # Delete option
            if st.button("🗑️ Remove Message"):
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history WHERE id = ?", (selected_msg['id'],))
                conn.commit()
                conn.close()
                st.experimental_rerun()

        # Show all messages
        st.markdown("### 💬 All Messages")
        for _, row in chat_df.iterrows():
            st.markdown(f"**{row['role']}**: {row['message']}")
    else:
        st.info("No chat history found.")

# --- Show Logs ---
st.subheader("📋 Retrieved Logs")
logs_df = fetch_logs()
st.dataframe(logs_df, use_container_width=True)

# --- Anomaly Detection ---
if not logs_df.empty:
    st.subheader("📊 AI Pattern & Anomaly Detection")
    analyzed_df, anomalies_df = detect_anomalies(logs_df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Anomalies Detected")
        if not anomalies_df.empty:
            st.dataframe(anomalies_df[['timestamp', 'level', 'message']], use_container_width=True)
        else:
            st.info("✨ No anomalies detected. Smooth sailing!")

    with col2:
        st.markdown("### 📈 Message Length Distribution")
        fig, ax = plt.subplots()
        ax.hist(analyzed_df['length'], bins=20, color='orchid', edgecolor='black')
        ax.set_xlabel("Log Message Length")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
