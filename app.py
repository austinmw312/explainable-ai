import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# Set page title and description
st.title("Explainable Sentiment Analysis")
st.markdown("Enter text below to analyze its sentiment and see why the model made its prediction.")

# Load model and tokenizer
@st.cache_resource
def load_model():
    with st.spinner('Loading model... (this may take a minute the first time)'):
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return tokenizer, model

tokenizer, model = load_model()

# Create text input area
user_input = st.text_area("Input text here:", height=100)

if st.button("Analyze"):
    if not user_input:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner('Analyzing...'):
            # Get base prediction
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            scores = scores.tolist()[0]
            
            predicted_class = scores.index(max(scores))
            confidence = max(scores)
            
            # Display results
            sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
            st.write(f"**Prediction:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2%}")
            
            try:
                # Get word importances by removing each word
                tokens = tokenizer.tokenize(user_input)
                importances = []
                base_score = scores[predicted_class]
                
                # Calculate importance of each word by removing it
                for i in range(len(tokens)):
                    # Create new text with word removed
                    new_tokens = tokens.copy()
                    new_tokens[i] = tokenizer.pad_token
                    new_text = tokenizer.convert_tokens_to_string(new_tokens)
                    
                    # Get prediction without this word
                    new_inputs = tokenizer(new_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        new_outputs = model(**new_inputs)
                        new_scores = torch.softmax(new_outputs.logits, dim=1)
                        new_score = new_scores[0][predicted_class].item()
                    
                    # Importance is how much score changes when word is removed
                    importance = abs(base_score - new_score)
                    importances.append(importance)
                
                # Normalize importances
                max_importance = max(importances)
                if max_importance > 0:
                    importances = [i/max_importance for i in importances]
                
                # Create visualization
                st.write("**Word Importance Visualization:**")
                fig, ax = plt.subplots(figsize=(10, 3))
                
                # Plot words with their importance
                for idx, (word, importance) in enumerate(zip(tokens, importances)):
                    color = 'red' if predicted_class == 1 else 'blue'
                    ax.text(idx, 0.5, word, alpha=min(importance * 2, 1.0),
                           color=color, ha='center', va='center')
                
                ax.set_xlim(-1, len(tokens))
                ax.set_ylim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Could not generate explanation visualization: {str(e)}") 