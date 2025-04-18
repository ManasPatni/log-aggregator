# log-aggregator
LogWise is a secure, AI-powered tool for analyzing system logs. Whether you're a developer, system administrator, or analyst, LogWise helps you detect anomalies, uncover patterns, and manage log history effortlessly.

## Contact Information
- LinkedIn: [Manas Patni][https://www.linkedin.com/in/manas-patni-0b1013235/)
- Email: manaspatni07@gmail.com 
- Phone: 9405957126
- project demo Youtube link: 

**📌 Overview **
LogWise allows you to upload local log files (.log, .txt, or .pdf), store them in a local SQLite database, and apply machine learning to detect anomalies in your logs. It features a user-friendly Streamlit interface and includes project history, chat memory, and visualizations — all while keeping your data private and local.

**⚙️ Features**
📂 Local File Upload: Supports .log, .txt, and .pdf formats
🧠 AI-Powered Anomaly Detection: Uses Isolation Forest to detect unusual patterns
📊 Interactive Visualizations: Histogram of log message lengths
🕒 Project History: Rename, delete, and manage past analysis sessions
💬 Chat Memory: Keeps a history of past AI-user interactions
🔐 Fully Offline: All data stays local — no external data transfer

**🛠️ Setup Instructions**
1. Clone the Repository
  git clone https://github.com/your-username/logwise.git
  cd logwise
2. Create a Virtual Environment (Optional but Recommended)
   python -m venv venv
   source venv/bin/activate     # On Windows: venv\Scripts\activate
3. Install Dependencies
   pip install -r requirements.txt
4. Run the Application
   streamlit run app.py


