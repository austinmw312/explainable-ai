import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import numpy as np
import plotly.graph_objects as go

class ModelAnalyzer:
    def __init__(self, model_name="gpt2"):
        """
        Initialize the model analyzer with a specific GPT-2 model variant
        model_name can be: 'gpt2', 'gpt2-medium', 'gpt2-large', or 'gpt2-xl'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer
        @st.cache_resource
        def load_model_and_tokenizer(name):
            tokenizer = GPT2Tokenizer.from_pretrained(name)
            model = GPT2LMHeadModel.from_pretrained(name, output_attentions=True)
            model = model.to(self.device)
            return model, tokenizer
        
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.model.eval()  # Set to evaluation mode
        
    def get_attention_patterns(self, text):
        """
        Get attention patterns for all layers and heads
        Returns: attention patterns, tokens
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self.device)
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Get attention patterns from all layers
        attention = outputs.attentions  # Tuple of tensors: (n_layers, batch, n_heads, seq_len, seq_len)
        
        # Convert tokens to readable format
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return attention, tokens
    
    def clean_token(self, token):
        """Clean token by removing special characters"""
        return token.replace('Ġ', ' ').strip()
    
    def visualize_attention(self, attention, tokens, layer_idx=0, head_idx=0):
        """
        Create attention pattern visualization for a specific layer and head
        """
        # Clean tokens for visualization
        clean_tokens = [self.clean_token(token) for token in tokens]
        
        # Get attention matrix for specified layer and head and move to CPU
        attention_matrix = attention[layer_idx][0][head_idx]
        if torch.cuda.is_available():
            attention_matrix = attention_matrix.cpu()
        attention_matrix = attention_matrix.detach().numpy()
        
        # Create heatmap with cleaned tokens
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=clean_tokens,
            y=clean_tokens,
            colorscale='Viridis'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Attention Pattern (Layer {layer_idx}, Head {head_idx})',
            xaxis_title="Target Tokens",
            yaxis_title="Source Tokens",
            width=800,
            height=800
        )
        
        return fig
    
    def analyze_text(self, text):
        """
        Perform complete analysis of input text
        """
        # Get attention patterns
        attention, tokens = self.get_attention_patterns(text)
        
        # Get model prediction
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Get next token prediction
        next_token_logits = logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        top_k = 5
        topk_probs, topk_indices = torch.topk(next_token_probs, top_k)
        
        # Clean predicted tokens
        predicted_tokens = [
            (self.clean_token(self.tokenizer.decode(idx.item())), prob.item())
            for idx, prob in zip(topk_indices, topk_probs)
        ]
        
        return {
            'attention': attention,
            'tokens': [self.clean_token(token) for token in tokens],
            'predicted_tokens': predicted_tokens
        }

# Example usage in Streamlit app:
def main():
    st.title("GPT-2 Model Analyzer")
    st.markdown("Analyze and visualize GPT-2's attention patterns and predictions")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Model selection
    model_name = st.selectbox(
        "Select GPT-2 model size:",
        ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
    )
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(model_name)
    
    # Text input
    text = st.text_area("Enter text to analyze:", "The quick brown fox jumps over the lazy dog")
    
    if st.button("Analyze"):
        with st.spinner("Analyzing text..."):
            # Get analysis results and store in session state
            st.session_state.analysis_results = analyzer.analyze_text(text)
    
    # Only show visualizations if we have results
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        
        # Display tokens
        st.subheader("Tokenization:")
        st.write(results['tokens'])
        
        # Display next token predictions
        st.subheader("Top 5 next token predictions:")
        for token, prob in results['predicted_tokens']:
            st.write(f"{token}: {prob:.3f}")
        
        # Attention visualization controls
        st.subheader("Attention Visualization")
        num_layers = len(results['attention'])
        num_heads = results['attention'][0].size(1)
        
        col1, col2 = st.columns(2)
        with col1:
            layer_idx = st.slider("Select layer:", 0, num_layers-1, 0, key='layer_slider')
        with col2:
            head_idx = st.slider("Select attention head:", 0, num_heads-1, 0, key='head_slider')
        
        # Display attention visualization
        fig = analyzer.visualize_attention(
            results['attention'],
            results['tokens'],
            layer_idx,
            head_idx
        )
        st.plotly_chart(fig)
        
        # After the attention visualization
        st.markdown("""
        ### How to Interpret the Attention Visualization:
        - The heatmap shows how each word (y-axis) attends to other words (x-axis)
        - Brighter colors indicate stronger attention
        - Different layers and heads capture different types of relationships:
          - Layer 0-3: Often capture basic patterns and local relationships
          - Middle layers: Mix of syntactic and semantic relationships
          - Final layers: More complex semantic relationships
        - Common Patterns:
          - Vertical line on first word: Many words attend to the sentence start for context
          - Diagonal lines: Words paying attention to nearby words
          - Vertical stripes: Words that are important globally
          - Bright spots: Related words or grammatically linked words

        Try this experiment: Compare attention patterns between sentences starting with "The" vs "A" vs no article!
        """)

if __name__ == "__main__":
    main() 