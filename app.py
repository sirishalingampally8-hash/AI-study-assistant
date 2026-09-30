"""
app.py - Streamlit UI for the AI Study Assistant.
Run with:  streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from rag_agent import StudyAssistant

st.set_page_config(page_title="AI Study Assistant", page_icon="📚", layout="wide")

DEFAULT_NOTES = "study_notes.txt"


# ---------------------------------------------------------------------- #
# Session setup
# ---------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model...")
def get_assistant() -> StudyAssistant:
    assistant = StudyAssistant()
    _ = assistant.embedder  # warm up the model once
    return assistant


assistant = get_assistant()

if "loaded" not in st.session_state:
    st.session_state.loaded = False
    st.session_state.mcqs = []
    st.session_state.submitted = False

# ---------------------------------------------------------------------- #
# Sidebar: notes
# ---------------------------------------------------------------------- #
with st.sidebar:
    st.header("📄 Your Notes")
    uploads = st.file_uploader(
        "Upload notes (.txt / .pdf)", type=["txt", "pdf"], accept_multiple_files=True
    )
    pasted = st.text_area("...or paste notes here", height=150)
    use_sample = st.checkbox("Include sample notes", value=not uploads and not pasted)

    if st.button("Build knowledge base", type="primary", use_container_width=True):
        assistant.clear()
        total = 0
        with st.spinner("Indexing notes..."):
            if use_sample and os.path.exists(DEFAULT_NOTES):
                total += assistant.add_file(DEFAULT_NOTES)
            for up in uploads or []:
                suffix = os.path.splitext(up.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = tmp.name
                total += assistant.add_file(tmp_path)
                os.remove(tmp_path)
            if pasted.strip():
                total += assistant.add_text(pasted, source="pasted notes")
        st.session_state.loaded = total > 0
        if total:
            st.success(f"Indexed {total} chunks.")
        else:
            st.warning("No notes found. Upload, paste, or tick sample notes.")

    st.divider()
    level = st.selectbox("Explanation level", ["Beginner", "Intermediate", "Advanced"])

# ---------------------------------------------------------------------- #
# Main area
# ---------------------------------------------------------------------- #
st.title("📚 AI Study Assistant")
st.caption("Explains notes, summarizes topics and generates MCQs.")

if not st.session_state.loaded:
    st.info("👈 Build the knowledge base from the sidebar to get started.")
    st.stop()

tab_explain, tab_summary, tab_mcq = st.tabs(["💡 Explain", "📝 Summarize", "❓ MCQ Quiz"])

# ---- Explain --------------------------------------------------------- #
with tab_explain:
    question = st.text_input("Ask about any concept in your notes",
                             placeholder="e.g. What is a deadlock?")
    if st.button("Explain", key="btn_explain") and question.strip():
        with st.spinner("Thinking..."):
            try:
                st.markdown(assistant.explain(question, level))
                with st.expander("Sources used"):
                    for hit in assistant.retrieve(question, k=3):
                        st.write(f"**{hit['source']}** (score {hit['score']:.2f})")
                        st.caption(hit["text"][:300] + "...")
            except Exception as e:
                st.error(f"Error: {e}")

# ---- Summarize ------------------------------------------------------- #
with tab_summary:
    topic = st.text_input("Topic (leave empty to summarize all notes)", key="sum_topic")
    style = st.radio("Style", ["Bullet points", "Short paragraph", "Exam revision sheet"],
                     horizontal=True)
    if st.button("Summarize", key="btn_sum"):
        with st.spinner("Summarizing..."):
            try:
                st.markdown(assistant.summarize(topic, style))
            except Exception as e:
                st.error(f"Error: {e}")

# ---- MCQs ------------------------------------------------------------ #
with tab_mcq:
    c1, c2, c3 = st.columns([3, 1, 1])
    mcq_topic = c1.text_input("Topic (optional)", key="mcq_topic")
    n_q = c2.number_input("Questions", 1, 15, 5)
    diff = c3.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    if st.button("Generate MCQs", key="btn_mcq"):
        with st.spinner("Writing questions..."):
            try:
                st.session_state.mcqs = assistant.generate_mcqs(mcq_topic, int(n_q), diff)
                st.session_state.submitted = False
                for i in range(15):
                    st.session_state.pop(f"q_{i}", None)
            except Exception as e:
                st.error(f"Error: {e}")

    mcqs = st.session_state.mcqs
    if mcqs:
        for i, q in enumerate(mcqs):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            choice = st.radio("Select an answer", q["options"], index=None,
                              key=f"q_{i}", label_visibility="collapsed")
            if st.session_state.submitted:
                correct = q["options"][q["answer"]]
                if choice == correct:
                    st.success("Correct ✅")
                else:
                    st.error(f"Wrong ❌ — correct answer: {correct}")
                if q["explanation"]:
                    st.caption(f"💬 {q['explanation']}")
            st.write("")

        if st.button("Submit answers", type="primary"):
            st.session_state.submitted = True
            st.rerun()

        if st.session_state.submitted:
            score = sum(
                st.session_state.get(f"q_{i}") == q["options"][q["answer"]]
                for i, q in enumerate(mcqs)
            )
            st.subheader(f"Score: {score} / {len(mcqs)}")
