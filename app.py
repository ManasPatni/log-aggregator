import streamlit as st
import pandas as pd
import sqlite3
import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import datetime

st.set_page_config(page_title="AI-Driven Log Aggregator", layout="wide")

# --- Database setup ---
DB_NAME = "logs.db"

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
    conn.commit()
    conn.close()

# --- Store logs in DB ---
def store_logs(df):
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("logs", conn, if_exists="append", index=False)
    conn.close()

# --- Read logs from DB ---
def fetch_logs():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM logs", conn)
    conn.close()
    return df

# --- Analyze logs with Isolation Forest ---
def detect_anomalies(df):
    if df.empty or len(df) < 10:
        return df, []

    df['length'] = df['message'].str.len()
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['length']])
    anomalies = df[df['anomaly'] == -1]
    return df, anomalies

# --- Parse uploaded log file ---
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
        except Exception as e:
            continue
    return pd.DataFrame(data)

# --- Streamlit UI ---
init_db()
st.title("🧠 AI-Driven Log Aggregator (Local)")

uploaded_file = st.file_uploader("Upload a log file (.log or .txt)", type=["log", "txt"])

if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        st.success("Logs successfully stored in local database.")
    else:
        st.error("No valid log entries found.")

st.subheader("📋 Retrieved Logs")
logs_df = fetch_logs()
st.dataframe(logs_df, use_container_width=True)

if not logs_df.empty:
    st.subheader("📊 Pattern Detection")
    analyzed_df, anomalies_df = detect_anomalies(logs_df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Anomalies Detected")
        st.dataframe(anomalies_df[['timestamp', 'level', 'message']], use_container_width=True)

    with col2:
        st.markdown("### Log Message Length Distribution")
        fig, ax = plt.subplots()
        ax.hist(analyzed_df['length'], bins=20, color='skyblue', edgecolor='black')
        st.pyplot(fig)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, SQLite, and Scikit-learn")
