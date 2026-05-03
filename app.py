import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "t5-small"   # ✅ FIXED
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ✅ CACHE CLEAR BUTTON (IMPORTANT)
st.button("Clear Cache", on_click=st.cache_resource.clear)

st.title("🧠 Multilingual Text Summarizer")

text = st.text_area("Enter your text")

if st.button("Summarize"):
    if text.strip() != "":
        input_text = "summarize: " + text

        inputs = tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        outputs = model.generate(
            inputs,
            max_length=120,
            num_beams=4,          # ✅ better output
            early_stopping=True
        )

        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.subheader("Summary:")
        st.write(summary)
