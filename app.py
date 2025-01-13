import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

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
                contributions = []
                base_scores = scores

                # Calculate importance of each word by removing it
                for i in range(len(tokens)):
                    new_tokens = tokens.copy()
                    new_tokens[i] = tokenizer.pad_token
                    new_text = tokenizer.convert_tokens_to_string(new_tokens)
                    
                    new_inputs = tokenizer(new_text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        new_outputs = model(**new_inputs)
                        new_scores = torch.softmax(new_outputs.logits, dim=1).tolist()[0]
                    
                    importance = abs(base_scores[1] - new_scores[1])
                    importances.append(importance)
                    contributes_positively = new_scores[1] < base_scores[1]
                    contributions.append(contributes_positively)

                # Normalize importances
                max_importance = max(importances)
                if max_importance > 0:
                    importances = [i/max_importance for i in importances]

                # Create custom HTML visualization
                st.markdown("""
                ### Visualization Results
                Words are colored based on their contribution to the sentiment, with scores showing their importance.
                
                <style>
                .word-container {
                    background-color: black;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                    line-height: 2.5;
                }
                .word-box {
                    display: inline-block;
                    margin: 0 10px;
                    text-align: center;
                }
                .score {
                    font-size: 12px;
                    margin-bottom: 5px;
                }
                .word {
                    font-size: 16px;
                    font-weight: bold;
                }
                </style>
                
                <div class="word-container">
                """, unsafe_allow_html=True)

                # Create word boxes with scores
                html_words = []
                for word, importance, is_positive in zip(tokens, importances, contributions):
                    color = "rgb(50, 205, 50)" if is_positive else "rgb(255, 50, 50)"
                    word_html = f"""
                    <div class="word-box">
                        <div class="score" style="color: {color}">{importance:.2f}</div>
                        <div class="word" style="color: {color}">{word}</div>
                    </div>
                    """
                    html_words.append(word_html)

                # Join words and close container
                st.markdown("".join(html_words) + "</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not generate explanation visualization: {str(e)}") 