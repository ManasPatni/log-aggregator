import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# --- Streamlit Page Config ---
st.set_page_config(page_title="LogWise 🔁️", layout="wide")
st.title("LogWise 🔁️ - AI-Powered Local Log Analyzer")
st.markdown("""
Welcome to **LogWise**!

✨ AI-powered local log analyzer with:
- 🔎 Pattern analysis
- 🚨 Anomaly detection
- 📂 Secure local storage
""")

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

# --- Logs Functions ---
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

# --- Chat Functions ---
def store_chat(role, message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (role, message) VALUES (?, ?)", (role, message))
    conn.commit()
    conn.close()

def fetch_chat():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql("SELECT * FROM chat_history ORDER BY id DESC LIMIT 20", conn)
    conn.close()
    return df[::-1]  # reverse to chronological order

def rename_chat_entry(chat_id, new_message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE chat_history SET message = ? WHERE id = ?", (new_message, chat_id))
    conn.commit()
    conn.close()

def delete_chat_entry(chat_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_history WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()

# --- Initialize ---
reset_db()
init_db()

# --- Upload Logs ---
st.markdown("### :file_folder: Upload a log file (.log or .txt) to get started:")
uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])

if uploaded_file:
    log_df = parse_log(uploaded_file)
    if not log_df.empty:
        store_logs(log_df)
        store_chat("assistant", "Logs successfully stored in the local database.")
        add_project("Logging Aggregator")

        st.subheader(":clipboard: Retrieved Logs")
        logs_df = fetch_logs()
        st.dataframe(logs_df, use_container_width=True)

        st.subheader(":bar_chart: AI Pattern & Anomaly Detection")
        analyzed_df, anomalies_df = detect_anomalies(logs_df)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### :mag: Anomalies Detected")
            if not anomalies_df.empty:
                st.dataframe(anomalies_df[['timestamp', 'level', 'message']], use_container_width=True)
            else:
                st.info(":sparkles: No anomalies detected. Smooth sailing!")
        with col2:
            st.markdown("### :chart_with_upwards_trend: Message Length Distribution")
            fig, ax = plt.subplots()
            ax.hist(analyzed_df['length'], bins=20, color='mediumseagreen', edgecolor='black')
            ax.set_xlabel("Log Message Length")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    else:
        store_chat("assistant", "No valid log entries found in the file.")
        st.error("No valid log entries found in the file.")

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("### 🕒 Project History")
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
                    do_share = col1.form_submit_button(":outbox_tray:", use_container_width=True)
                    do_rename = col2.form_submit_button(":pencil2:", use_container_width=True)
                    do_archive = col3.form_submit_button(":inbox_tray:", use_container_width=True)
                    do_delete = col4.form_submit_button(":wastebasket:", use_container_width=True)

                    if do_rename:
                        rename_project(row["id"], new_title)
                        st.success("Renamed successfully!")
                        st.experimental_rerun()
                    elif do_archive:
                        st.info(":inbox_tray: Archive not implemented yet.")
                    elif do_delete:
                        delete_project(row["id"])
                        st.warning("Deleted.")
                        st.experimental_rerun()
                    elif do_share:
                        st.info(":outbox_tray: Share feature coming soon!")

    st.markdown("---")
    st.markdown("### 💬 Chat History")
    chat_df = fetch_chat()
    for _, row in chat_df.iterrows():
        with st.expander(f"{'👤' if row['role'] == 'user' else '🤖'} {row['role'].capitalize()} - {row['message']}", expanded=False):
            form_key = f"chat_form_{row['id']}"
            with st.form(form_key):
                new_message = st.text_area("Rename", value=row['message'], key=f"chat_input_{row['id']}")

                col1, col2, col3, col4 = st.columns(4)
                do_share = col1.form_submit_button(":outbox_tray:", use_container_width=True)
                do_rename = col2.form_submit_button(":pencil2:", use_container_width=True)
                do_archive = col3.form_submit_button(":inbox_tray:", use_container_width=True)
                do_delete = col4.form_submit_button(":wastebasket:", use_container_width=True)

                if do_rename:
                    rename_chat_entry(row["id"], new_message)
                    st.success("Chat message renamed!")
                    st.experimental_rerun()
                elif do_delete:
                    delete_chat_entry(row["id"])
                    st.warning("Chat message deleted.")
                    st.experimental_rerun()
                elif do_archive:
                    st.info(":inbox_tray: Archive feature not implemented yet.")
                elif do_share:
                    st.info(":outbox_tray: Share feature coming soon!")
