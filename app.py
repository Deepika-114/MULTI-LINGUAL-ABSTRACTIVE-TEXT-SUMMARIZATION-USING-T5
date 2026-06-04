import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pdfplumber
import pytesseract
from pdf2image import convert_from_bytes
import re
import torch

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Multilingual AI Summarizer", layout="centered")

# ---------- SESSION STATE ----------
if "started" not in st.session_state:
    st.session_state.started = False

# ---------- DEVICE ----------
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- MODEL ----------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return tokenizer, model

# ---------- CLEAN TEXT ----------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------- CHUNK TEXT ----------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# ---------- SUMMARIZE ----------
def summarize_text(text, tokenizer, model, max_len):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        input_text = "summarize: " + chunk

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len,
            num_beams=4,
            early_stopping=True
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# ---------- PDF EXTRACTION ----------
def extract_text_from_pdf(uploaded_file):
    text = ""

    # TRY pdfplumber first
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    except:
        pass

    # OCR fallback
    if len(text.strip()) < 50:
        st.warning("⚠️ Scanned PDF detected → Using OCR...")

        uploaded_file.seek(0)
        images = convert_from_bytes(uploaded_file.read())

        for img in images:
            text += pytesseract.image_to_string(img, config="--psm 6")

    return text

# ---------- WELCOME ----------
if not st.session_state.started:
    st.markdown("<h1 style='text-align:center;'>👋 Welcome</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Multilingual AI PDF Summarizer (mT5)</p>", unsafe_allow_html=True)

    if st.button("🚀 Start"):
        st.session_state.started = True
        st.rerun()

# ---------- MAIN ----------
else:
    st.title("🧠 Multilingual AI Summarizer")

    # BEST MODEL FOR YOU
    model_option = st.selectbox(
        "Choose Model",
        [
            "csebuetnlp/mT5_multilingual_XLSum",
            "google/mt5-small",
            "google/mt5-base"
        ]
    )

    tokenizer, model = load_model(model_option)

    st.divider()

    # FILE UPLOAD
    uploaded_file = st.file_uploader("📄 Upload PDF or TXT", type=["pdf", "txt"])

    text = ""

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")

    # TEXT INPUT
    text_input = st.text_area("✍️ Input Text", value=text, height=220)

    # SETTINGS
    summary_length = st.slider("Summary Length", 50, 250, 120)

    # WORD COUNT
    if text_input:
        st.write(f"📊 Words: {len(text_input.split())}")

    # SUMMARIZE BUTTON
    if st.button("✨ Summarize"):
        if len(text_input.strip()) > 20:

            with st.spinner("Generating summary... ⏳"):
                clean = clean_text(text_input)

                summary = summarize_text(
                    clean,
                    tokenizer,
                    model,
                    summary_length
                )

            st.success("✅ Summary Ready")
            st.subheader("📌 Summary")
            st.write(summary)

            st.write(f"📊 Summary Words: {len(summary.split())}")

            st.download_button(
                "📥 Download Summary",
                summary,
                file_name="summary.txt"
            )

        else:
            st.warning("⚠️ Please enter more text")

    # DEBUG INFO
    st.write("🔍 Extracted Text Length:", len(text))

    # CONTROLS
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Clear Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared")

    with col2:
        if st.button("⬅️ Reset"):
            st.session_state.started = False
            st.rerun()
