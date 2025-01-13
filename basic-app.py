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
            try:
                # Tokenize input
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
                
                # Get embeddings and enable gradient tracking
                embeddings = model.get_input_embeddings()
                input_ids = inputs['input_ids']
                embedded = embeddings(input_ids)
                embedded.retain_grad()
                
                # Get prediction
                outputs = model(inputs_embeds=embedded, attention_mask=inputs['attention_mask'])
                scores = torch.softmax(outputs.logits, dim=1)
                predicted_class = scores.argmax(-1).item()
                confidence = scores[0, predicted_class].item()
                
                # Get gradients for predicted class
                scores[0, predicted_class].backward()
                
                # Calculate importance scores
                importance = embedded.grad.abs().mean(dim=-1)[0]
                # Normalize importance scores
                importance = importance / importance.max()
                
                # Get tokens
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # Determine contribution direction (positive/negative)
                with torch.no_grad():
                    # Get base positive score
                    base_positive = scores[0, 1].item()
                    
                    # Check each token's contribution
                    contributions = []
                    for i in range(len(tokens)):
                        # Zero out embedding for this token
                        temp_embedded = embedded.clone()
                        temp_embedded[0, i] = 0
                        # Get new prediction
                        temp_output = model(inputs_embeds=temp_embedded, 
                                         attention_mask=inputs['attention_mask'])
                        temp_scores = torch.softmax(temp_output.logits, dim=1)
                        # If removing token decreases positive score, it was contributing positively
                        contributes_positively = temp_scores[0, 1].item() < base_positive
                        contributions.append(contributes_positively)
                
                # Display results
                sentiment = "POSITIVE" if predicted_class == 1 else "NEGATIVE"
                st.write(f"**Prediction:** {sentiment}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Create visualization
                st.markdown("""
                ### Visualization Results
                Words are colored based on their contribution to the sentiment, with scores showing their importance.
                
                <style>
                .word-container {
                    padding: 10px;
                    margin: 10px 0;
                    line-height: 5.0;
                }
                .word-box {
                    display: inline-block;
                    margin: 0 10px 20px 10px;
                    text-align: center;
                    vertical-align: top;
                }
                .score {
                    font-size: 12px;
                    margin-bottom: 2px;
                }
                .word {
                    font-size: 16px;
                    font-weight: bold;
                    margin-top: 2px;
                }
                </style>
                
                <div class="word-container">
                """, unsafe_allow_html=True)

                # Create word boxes with scores
                html_words = []
                for word, imp, is_positive in zip(tokens, importance, contributions):
                    if word not in ['[CLS]', '[SEP]', '[PAD]']:  # Skip special tokens
                        color = "rgb(50, 205, 50)" if is_positive else "rgb(255, 50, 50)"
                        word_html = f"""
                        <div class="word-box">
                            <div class="score" style="color: {color}">{imp:.2f}</div>
                            <div class="word" style="color: {color}">{word}</div>
                        </div>
                        """
                        html_words.append(word_html)

                # Join words and close container
                st.markdown("".join(html_words) + "</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Could not generate explanation visualization: {str(e)}") 