import sqlite3
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="LogWise 👁️‍🗨️", layout="wide")
st.title("👁️‍🗨️ LogWise — AI-Powered Local Log Analyzer")
st.markdown("Welcome to **LogWise**! Upload your log file and watch it work: 🧠 **pattern analysis**, 🚨 **anomaly detection**, and 🗄️ **local storage** — no cloud involved!")

DB_NAME = "logs.db"

# --- DB Reset and Init ---
def reset_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS logs")
    cursor.execute("DROP TABLE IF EXISTS chat_history")
    conn.commit()
    conn.close()

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        level TEXT,
        message TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        message TEXT)''')
    conn.commit()
    conn.close()

# --- Data Ops ---
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
                data.append({'timestamp': parts[0], 'level': parts[1], 'message': parts[2]})
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

# --- Init ---
reset_db()
init_db()

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

# --- Sidebar: Copilot-style History ---
with st.sidebar:
    st.markdown("### 📁 Project History")

    # You can dynamically fetch and render projects here
    project_groups = {
        "Today": ["Logging Aggregator"],
        "Yesterday": ["AI-driven MCQ Generator"],
        "Previous 7 Days": ["learning python (hidevs)", "Beginner Python Questions"],
        "Previous 30 Days": ["Math Problem Assistant", "language translator project"]
    }

    for section, projects in project_groups.items():
        st.markdown(f"#### {section}")
        for proj in projects:
            with st.expander(proj, expanded=False):
                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                with col1: st.button("📤", key=f"{proj}-share", help="Share")
                with col2: st.button("✏️", key=f"{proj}-rename", help="Rename")
                with col3: st.button("📦", key=f"{proj}-archive", help="Archive")
                with col4: st.button("🗑️", key=f"{proj}-delete", help="Delete")

# --- Logs Table ---
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
