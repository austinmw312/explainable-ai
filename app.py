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
- **Opacity**: 
  - More solid = Word has stronger influence
  - More faded = Word has weaker influence
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
                The words below are colored based on their contribution to the sentiment and their opacity shows how strong that contribution is.
                """)
                
                # Create visualization with black background
                fig, ax = plt.subplots(figsize=(10, 3), facecolor='black')
                ax.set_facecolor('black')
                
                # Plot words with their importance
                for idx, (word, importance, is_positive) in enumerate(zip(tokens, importances, contributions)):
                    # Green if word contributes positively, red if negatively
                    color = 'green' if is_positive else 'red'
                    # Increase minimum opacity to 0.7, scale remaining 0.3 by importance
                    opacity = 0.7 + (0.3 * importance)
                    ax.text(idx, 0.5, word, alpha=opacity,
                           color=color, ha='center', va='center')
                
                ax.set_xlim(-1, len(tokens))
                ax.set_ylim(0, 1)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Could not generate explanation visualization: {str(e)}") 

# After creating the visualization, add an example
if st.button("Show Example"):
    st.markdown("""
    **Example**: In the sentence "The movie was terrible but the acting was brilliant":
    - "terrible" would be red and solid (strong negative)
    - "brilliant" would be green and solid (strong positive)
    - "the" and "was" might be faded (less important to sentiment)
    """) 