import streamlit as st
from transformers import pipeline
import random

# -------------------------------
# Question & Option Generator
# -------------------------------
@st.cache_resource
def load_qg_model():
    return pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

qg_pipeline = load_qg_model()

def generate_mcq_from_text(text):
    sentences = text.split('.')
    if len(sentences) < 2:
        return None
    context = random.choice(sentences).strip()
    if len(context.split()) < 2:
        return None
    highlight = random.choice(context.split())
    input_text = f"highlight: {highlight} context: {context}"

    try:
        output = qg_pipeline(input_text)[0]['generated_text']
    except Exception as e:
        return None
    
    correct_answer = highlight
    distractors = list(set([
        highlight[::-1].capitalize(),
        highlight.upper(),
        highlight.lower(),
        highlight + "ing"
    ]))
    options = random.sample([correct_answer] + distractors[:3], 4)

    return {
        "question": output,
        "options": options,
        "answer": correct_answer
    }

# -------------------------------
# Adaptive Difficulty Manager
# -------------------------------
class DifficultyManager:
    def __init__(self):
        self.score = 0
        self.total_questions = 0

    def record_result(self, correct):
        self.total_questions += 1
        if correct:
            self.score += 1

    def get_difficulty(self):
        if self.total_questions == 0:
            return "Easy"
        accuracy = self.score / self.total_questions
        if accuracy > 0.75:
            return "Hard"
        elif accuracy > 0.5:
            return "Medium"
        return "Easy"

# -------------------------------
# Streamlit UI Layout Update
# -------------------------------
st.set_page_config(page_title="Adaptive MCQ Generator", layout="wide")
st.title("🎯 AI-Driven Adaptive MCQ Generator")

if 'manager' not in st.session_state:
    st.session_state.manager = DifficultyManager()

left, right = st.columns([2, 2])

# --- User Input (Right Side) ---
with right:
    st.header("🧑‍🎓 User Input")
    text_input = st.text_area("📘 Paste Educational Content:", height=200)

    if st.button("Generate MCQ"):
        if text_input.strip():
            mcq = generate_mcq_from_text(text_input)
            if mcq:
                st.session_state.mcq = mcq
            else:
                st.warning("⚠️ Could not generate a question. Try longer/more diverse text.")

# --- Bot Output (Left Side) ---
with left:
    st.header("🤖 AI Assistant")
    if 'mcq' in st.session_state:
        mcq = st.session_state.mcq
        st.markdown(f"**❓ Question:** {mcq['question']}")
        selected_option = st.radio("🔘 Choose your answer:", mcq['options'], key="mcq_radio")

        if st.button("Submit Answer"):
            correct = selected_option == mcq['answer']
            st.session_state.manager.record_result(correct)
            if correct:
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Incorrect. The correct answer is: **{mcq['answer']}**")

            st.info(f"📊 Current Difficulty Level: **{st.session_state.manager.get_difficulty()}**")
