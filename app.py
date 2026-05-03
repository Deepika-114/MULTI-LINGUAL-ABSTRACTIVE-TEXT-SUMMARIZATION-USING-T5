import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import PyPDF2

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="AI Summarizer", layout="centered")

# ---------- SESSION STATE ----------
if "started" not in st.session_state:
    st.session_state.started = False

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ---------- WELCOME SCREEN ----------
if not st.session_state.started:
    st.markdown("<h1 style='text-align:center;'>👋 Hello, Welcome!</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px;'>"
        "This AI-powered app helps you summarize long text and documents into short, meaningful insights instantly."
        "</p>",
        unsafe_allow_html=True
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    if st.button("🚀 Let's Start"):
        st.session_state.started = True
        st.rerun()

# ---------- MAIN APP ----------
else:
    tokenizer, model = load_model()

    st.markdown("<h1 style='text-align:center;'>🧠 AI Text Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload or paste text to summarize</p>", unsafe_allow_html=True)

    st.divider()

    # ---------- FILE UPLOAD ----------
    uploaded_file = st.file_uploader("📄 Upload a file (PDF or TXT)", type=["pdf", "txt"])

    text = ""

    # ---------- READ FILE ----------
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            text = uploaded_file.read().decode("utf-8")

    # ---------- TEXT INPUT ----------
    text_input = st.text_area("✍️ Or paste your text here", value=text, height=200)

    # ---------- SUMMARIZE ----------
    if st.button("✨ Summarize"):
        if text_input.strip() != "":
            with st.spinner("Generating summary... ⏳"):
                input_text = "summarize: " + text_input

                inputs = tokenizer.encode(
                    input_text,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                )

                outputs = model.generate(
                    inputs,
                    max_length=120,
                    num_beams=4,
                    early_stopping=True
                )

                summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            st.success("✅ Summary Generated")
            st.subheader("📌 Summary:")
            st.write(summary)
        else:
            st.warning("⚠️ Please enter or upload text first")

    # ---------- EXTRA BUTTONS ----------
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Clear Cache"):
            st.cache_resource.clear()

    with col2:
        if st.button("⬅️ Back"):
            st.session_state.started = False
            st.rerun()
