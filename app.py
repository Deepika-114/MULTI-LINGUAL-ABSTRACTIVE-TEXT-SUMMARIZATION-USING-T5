import streamlit as st
from transformers import pipeline

st.title("🌍 Multilingual Text Summarizer (T5)")

@st.cache_resource
def load_model():
    return pipeline("summarization", model="t5-small")

summarizer = load_model()

text = st.text_area("Enter text (any language):")

if st.button("Summarize"):
    if text:
        result = summarizer(text, max_length=120, min_length=30, do_sample=False)
        st.success(result[0]['summary_text'])
    else:
        st.warning("Please enter text")
