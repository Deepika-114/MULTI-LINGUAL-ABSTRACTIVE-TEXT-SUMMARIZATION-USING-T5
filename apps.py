import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.title("🌍 Multilingual Text Summarizer (T5)")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter text:")

if st.button("Summarize"):
    if text:
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = model.generate(inputs["input_ids"], max_length=120, min_length=30)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.success(output)
    else:
        st.warning("Please enter text")
