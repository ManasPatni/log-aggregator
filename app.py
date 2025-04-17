import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="LogWise 🔁️", layout="wide")
st.title("🔁️ LogWise — AI-Powered Local Log Analyzer")
st.markdown("Welcome to **LogWise**! Upload your log file and watch it work: 🧐 **pattern analysis**, 🚨 **anomaly detection**, and 🗔️ **local storage** — no cloud involved!")

DB_NAME = "logs.db"

# --- DB Setup ---
def reset_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS logs")
    cursor.execute("DROP TABLE IF EXISTS chat_history")
    cursor.execute("DROP TABLE IF EXISTS projects")
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
    cursor.execute('''CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

# --- Project Actions ---
def add_project(title):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO projects (title) VALUES (?)", (title,))
    conn.commit()
    conn.close()

def fetch_projects():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM projects ORDER BY created_at DESC", conn)
    conn.close()
    return df

def delete_project(project_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.commit()
    conn.close()

def rename_project(project_id, new_title):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE projects SET title = ? WHERE id = ?", (new_title, project_id))
    conn.commit()
    conn.close()

# --- Logs/Chat Functions ---
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

# --- INIT ---
reset_db()
init_db()

# --- Upload Logs ---
st.chat_message("user").markdown("📂 Upload a log file (.log or .txt) to get started:")
uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])

if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        message = "✅ Logs successfully stored in the local database."
        st.chat_message("assistant").success(message)
        store_chat("assistant", message)
        add_project("Logging Aggregator")
    else:
        message = "⚠️ No valid log entries found in the file."
        st.chat_message("assistant").error(message)
        store_chat("assistant", message)

# --- Sidebar Project History UI ---
with st.sidebar:
    st.markdown("### 🕓 Project History")
    projects = fetch_projects()

    def categorize_date(date_str):
        created = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        if created.date() == now.date():
            return "Today"
        elif created.date() == (now - timedelta(days=1)).date():
            return "Yesterday"
        elif created.date() >= (now - timedelta(days=7)).date():
            return "Previous 7 Days"
        return "Previous 30 Days"

    categorized = {}
    for _, row in projects.iterrows():
        category = categorize_date(row["created_at"])
        categorized.setdefault(category, []).append(row)

    for section, rows in categorized.items():
        st.markdown(f"#### {section}")
        for row in rows:
            with st.expander(row["title"], expanded=False):
                form_key = f"form_{row['id']}"
                with st.form(form_key):
                    new_title = st.text_input("Rename", value=row["title"], key=f"rename_input_{row['id']}")
                    col1, col2, col3, col4 = st.columns(4)
                    do_share = col1.form_submit_button("📤", use_container_width=True)
                    do_rename = col2.form_submit_button("✏️", use_container_width=True)
                    do_archive = col3.form_submit_button("🛆", use_container_width=True)
                    do_delete = col4.form_submit_button("🗑️", use_container_width=True)

                    if do_rename:
                        rename_project(row["id"], new_title)
                        st.success("Renamed successfully!")
                        st.experimental_rerun()
                    elif do_archive:
                        st.info("🛆 Archive not implemented yet.")
                    elif do_delete:
                        delete_project(row["id"])
                        st.warning("Deleted.")
                        st.experimental_rerun()
                    elif do_share:
                        st.info("📢 Share feature coming soon!")
# --- Upload Logs ---
st.chat_message("user").markdown("📂 Upload a log file (.log or .txt) to get started:")
uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])

if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        message = "✅ Logs successfully stored in the local database."
        st.chat_message("assistant").success(message)
        store_chat("assistant", message)
        add_project("Logging Aggregator")

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

    else:
        message = "⚠️ No valid log entries found in the file."
        st.chat_message("assistant").error(message)
        store_chat("assistant", message)


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
