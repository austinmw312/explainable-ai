import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.write("Debug: Starting app")  # Debug print

# Set page title and description
st.title("Explainable Sentiment Analysis")
st.markdown("Enter text below to analyze its sentiment and see why the model made its prediction.")

st.write("Debug: About to load model")  # Debug print

# Load model and tokenizer (this may take a few moments)
@st.cache_resource
def load_model():
    st.write("Debug: Inside load_model")  # Debug print
    with st.spinner('Loading model... (this may take a minute the first time)'):
        # Using a smaller model pre-trained for sentiment
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model

tokenizer, model = load_model()

st.write("Debug: Model loaded")  # Debug print

# Create text input area
user_input = st.text_area("Input text here:", height=100)

if st.button("Analyze"):
    if not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            # Tokenize and predict
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            
            # Get prediction
            predicted_class = torch.argmax(outputs.logits).item()
            confidence = torch.softmax(outputs.logits, dim=1).max().item()
            
            # Display results
            sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            st.write(f"**Prediction:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2%}") 