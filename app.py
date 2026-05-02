import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="T5 Summarizer", layout="centered")

st.title("🌍 Multilingual Abstractive Text Summarization using T5")

st.write("Enter text in any language and get a summarized version.")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="google/mt5-small")

summarizer = load_model()

text = st.text_area("Enter your text here:", height=200)

if st.button("Summarize"):
    if text.strip() != "":
        with st.spinner("Generating summary..."):
            result = summarizer(text, max_length=120, min_length=30, do_sample=False)
            st.success("Summary:")
            st.write(result[0]['summary_text'])
    else:
        st.warning("Please enter some text!")
