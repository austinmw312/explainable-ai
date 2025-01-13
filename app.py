import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

# Set page title and description
st.title("Explainable Sentiment Analysis")
st.markdown("Enter text below to analyze its sentiment and see why the model made its prediction.")
st.markdown("""
### How to interpret the visualization:
- **Colors**: 
  - 🟢 Green words contribute to positive sentiment
  - 🔴 Red words contribute to negative sentiment
- **Scores**: 
  - Numbers above words show importance (0.00 to 1.00)
  - Higher scores = Stronger influence on sentiment

**Example**: In the sentence "The movie was terrible but the acting was brilliant":
- "terrible" would be red with a high score (e.g., 0.85) showing strong negative influence
- "brilliant" would be green with a high score (e.g., 0.90) showing strong positive influence
- "the" and "was" would have low scores (e.g., 0.10) showing minimal influence
""")

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
                contributions = []  # Track if word contributes positively or negatively
                base_scores = scores  # Save both positive and negative scores
                
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
                        new_scores = torch.softmax(new_outputs.logits, dim=1).tolist()[0]
                    
                    # Importance is how much scores change when word is removed
                    importance = abs(base_scores[1] - new_scores[1])  # Use positive class score change
                    importances.append(importance)
                    
                    # If removing word decreases positive score, it was contributing positively
                    contributes_positively = new_scores[1] < base_scores[1]
                    contributions.append(contributes_positively)
                
                # Normalize importances
                max_importance = max(importances)
                if max_importance > 0:
                    importances = [i/max_importance for i in importances]
                
                # Before creating the visualization
                st.markdown("""
                ### Visualization Results
                The words below are colored based on their contribution to the sentiment and their score shows how strong that contribution is.
                """)
                
                # Create visualization with black background
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')  # Made taller for scores
                ax.set_facecolor('black')
                
                # Plot words with their importance
                for idx, (word, importance, is_positive) in enumerate(zip(tokens, importances, contributions)):
                    # Brighter green for positive, red for negative
                    color = 'limegreen' if is_positive else 'red'
                    
                    # Plot word at full opacity with larger font size
                    ax.text(idx, 0.35, word, alpha=1.0,
                           color=color, ha='center', va='center', fontsize=14)
                    
                    # Plot importance score above word
                    score = f"{importance:.2f}"
                    ax.text(idx, 0.55, score, alpha=1.0,
                           color=color, ha='center', va='center', fontsize=10)
                
                ax.set_xlim(-1, len(tokens))
                ax.set_ylim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Could not generate explanation visualization: {str(e)}") 