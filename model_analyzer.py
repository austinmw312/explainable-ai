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
    
    def visualize_attention(self, attention, tokens, layer_idx=0, head_idx=0):
        """
        Create attention pattern visualization for a specific layer and head
        """
        # Get attention matrix for specified layer and head
        attention_matrix = attention[layer_idx][0][head_idx].cpu().numpy()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
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
        
        predicted_tokens = [
            (self.tokenizer.decode(idx.item()), prob.item())
            for idx, prob in zip(topk_indices, topk_probs)
        ]
        
        return {
            'attention': attention,
            'tokens': tokens,
            'predicted_tokens': predicted_tokens
        }

# Example usage in Streamlit app:
def main():
    st.title("GPT-2 Model Analyzer")
    st.markdown("Analyze and visualize GPT-2's attention patterns and predictions")
    
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
            # Get analysis results
            results = analyzer.analyze_text(text)
            
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
                layer_idx = st.slider("Select layer:", 0, num_layers-1, 0)
            with col2:
                head_idx = st.slider("Select attention head:", 0, num_heads-1, 0)
            
            # Display attention visualization
            fig = analyzer.visualize_attention(
                results['attention'],
                results['tokens'],
                layer_idx,
                head_idx
            )
            st.plotly_chart(fig)

if __name__ == "__main__":
    main() 