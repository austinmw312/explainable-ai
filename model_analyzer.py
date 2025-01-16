import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Define color schemes
ATTENTION_COLORSCALE = 'Viridis' 
ACTIVATION_COLORSCALE = 'Viridis'
INFLUENCE_COLORSCALE = 'Viridis'
PREDICTION_COLORS = px.colors.qualitative.Set1

# Set page title and other configurations
st.set_page_config(
    page_title="GPT-2 Model Analyzer",
    page_icon="🤖",
    layout="wide"
)

class ModelAnalyzer:
    def __init__(self, model_name="gpt2"):
        """Initialize the model analyzer with GPT-2 base model"""
        self.device = "cpu"  # Force CPU for deployment
        
        # Load model and tokenizer with memory management
        @st.cache_resource(max_entries=1)
        def load_model_and_tokenizer(name):
            try:
                tokenizer = GPT2Tokenizer.from_pretrained(name)
                model = GPT2LMHeadModel.from_pretrained(name, output_attentions=True)
                model = model.to(self.device)
                return model, tokenizer
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None, None
        
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        if self.model is not None:
            self.model.eval()

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
        Note: layer_idx 0 is the first transformer layer (after embeddings)
        """
        # Clean tokens for visualization
        clean_tokens = [self.clean_token(token) for token in tokens]
        
        # Get attention matrix for specified layer and move to CPU
        attention_matrix = attention[layer_idx][0][head_idx]
        if torch.cuda.is_available():
            attention_matrix = attention_matrix.cpu()
        attention_matrix = attention_matrix.detach().numpy()
        
        # Create heatmap with cleaned tokens
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=clean_tokens,
            y=clean_tokens,
            colorscale=ATTENTION_COLORSCALE
        ))
        
        # Update layout with clear layer numbering
        fig.update_layout(
            title=f'Attention Pattern (Layer {layer_idx + 1}, Head {head_idx})',  # Add 1 to match activation numbering
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
        # Get attention patterns and hidden states
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True
            )
        
        attention = outputs.attentions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Get next token prediction
        next_token_logits = outputs.logits[0, -1, :]
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
            'predicted_tokens': predicted_tokens,
            'hidden_states': outputs.hidden_states
        }
    
    def visualize_token_influence(self, text, tokens):
        """Visualize how each token influences the final prediction"""
        # Get embeddings and create tensor that requires gradient
        embeddings = self.model.get_input_embeddings()
        tokens_tensor = self.tokenizer(text, return_tensors="pt")['input_ids'].to(self.device)
        token_embeddings = embeddings(tokens_tensor)
        token_embeddings.retain_grad()
        
        # Forward pass with gradient tracking
        outputs = self.model(inputs_embeds=token_embeddings)
        # Get gradients for the last token prediction
        outputs.logits[:, -1, :].sum().backward()
        
        # Calculate influence scores
        influence = token_embeddings.grad.abs().mean(dim=-1)[0].cpu()
        influence_values = influence.detach().numpy()
        
        # Create influence bar chart
        fig = go.Figure(data=go.Bar(
            x=[self.clean_token(t) for t in tokens],
            y=influence_values,
            marker=dict(
                color=influence_values,
                colorscale=INFLUENCE_COLORSCALE
            )
        ))
        
        # Update layout
        fig.update_layout(
            title='Token Influence on Final Prediction',
            xaxis_title="Tokens",
            yaxis_title="Influence Score",
            width=800,
            height=400
        )
        
        return fig
    
    def visualize_layer_activations(self, text, tokens):
        """Visualize activation patterns across all layers"""
        try:
            # Get model outputs with hidden states
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
                # Stack all hidden states and move to CPU
                hidden_states = torch.stack(outputs.hidden_states).to('cpu')
                
                # Print shape information for debugging
                print(f"Hidden states shape: {hidden_states.shape}")
                
                # Calculate activation magnitudes and squeeze out batch dimension
                activation_magnitudes = hidden_states.abs().mean(dim=-1).squeeze(1).detach().numpy()
                print(f"Activation magnitudes shape: {activation_magnitudes.shape}")
                print(f"Activation range: {activation_magnitudes.min():.3f} to {activation_magnitudes.max():.3f}")
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=activation_magnitudes,
                    x=[self.clean_token(t) for t in tokens],
                    y=[
                        "Embedding Layer" if i == 0 
                        else f"Layer {i}" if i < 12
                        else "Output Layer" 
                        for i in range(hidden_states.shape[0])
                    ],
                    colorscale=ACTIVATION_COLORSCALE,
                    showscale=True,
                    zmin=0  # Force minimum value to be 0 for better contrast
                ))
                
                # Update layout
                fig.update_layout(
                    title='Layer Activation Patterns',
                    xaxis_title="Tokens",
                    yaxis_title="Model Layers",
                    width=800,
                    height=600
                )
                
                return fig
        except Exception as e:
            print(f"Error in layer activation visualization: {str(e)}")
            raise e
    
    def visualize_prediction_flow(self, text, tokens):
        """Visualize how prediction probabilities evolve through layers"""
        try:
            # Get model outputs with hidden states
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                # Get all hidden states
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True
                )
                
                # Track probabilities through layers
                layer_probs = []
                for hidden_state in outputs.hidden_states:
                    # Project hidden state to vocabulary
                    logits = self.model.lm_head(hidden_state)
                    # Get probabilities for next token
                    probs = torch.softmax(logits[0, -1, :], dim=-1)
                    # Get top 5 tokens and their probabilities
                    top_probs, top_indices = probs.topk(5)
                    layer_probs.append(top_probs.cpu().numpy())
                
                # Create line plot
                fig = go.Figure()
                
                # Get token labels for legend
                top_tokens = [self.clean_token(self.tokenizer.decode(idx.item())) 
                             for idx in top_indices]
                
                # Add a line for each top token
                for i in range(5):
                    fig.add_trace(go.Scatter(
                        x=list(range(len(layer_probs))),
                        y=[probs[i] for probs in layer_probs],
                        name=f'"{top_tokens[i]}"',
                        mode='lines+markers',
                        line=dict(color=PREDICTION_COLORS[i])
                    ))
                
                # Update layout
                fig.update_layout(
                    title='Prediction Probability Flow Through Layers',
                    xaxis_title="Layer",
                    yaxis_title="Probability",
                    xaxis=dict(
                        ticktext=["Embedding"] + [f"Layer {i}" for i in range(1, 12)] + ["Output"],
                        tickvals=list(range(13))
                    ),
                    width=800,
                    height=400,
                    yaxis_range=[0, 1]
                )
                
                return fig
                
        except Exception as e:
            print(f"Error in prediction flow visualization: {str(e)}")
            raise e

def main():
    st.title("GPT-2 Model Analyzer")
    st.markdown("""
    Analyze and visualize GPT-2's attention patterns and predictions.
    
    Note: This demo uses the base GPT-2 model for performance reasons.
    For larger models and more features, consider running locally.
    """)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # Simplified model selection for deployment
    model_name = "gpt2"  # Only use base model
    
    try:
        # Initialize analyzer
        analyzer = ModelAnalyzer(model_name)
        
        # Text input with length limit
        default_text = (
            "The lighthouse keeper watched the approaching storm. The waves grew larger with each passing hour until they reached a"
        )
        text = st.text_area(
            "Enter text to analyze (max 500 characters):", 
            default_text,
            max_chars=500
        )
        
        if st.button("Analyze"):
            # Strip trailing whitespace from input text
            text = text.rstrip()
            
            if len(text) > 500:
                st.warning("Please enter shorter text (max 500 characters)")
            else:
                with st.spinner("Analyzing text... (this may take a moment)"):
                    try:
                        st.session_state.analysis_results = analyzer.analyze_text(text)
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            st.error("Sorry, the model ran out of memory. Try with shorter text.")
                        else:
                            st.error(f"An error occurred: {str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
        
        # Only show visualizations if we have results
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Display tokens
            with st.expander("Show Tokenization"):
                st.write(results['tokens'])
            
            # Display next token predictions
            st.subheader("Top 5 next token predictions:")
            df = pd.DataFrame(
                results['predicted_tokens'],
                columns=['Token', 'Probability']
            )
            df.index = range(1, len(df) + 1)
            df['Probability'] = df['Probability'].map('{:.1%}'.format)
            st.table(df)
            
            # Prediction Flow
            st.subheader("Prediction Probability Flow")
            st.markdown("""
            This visualization shows how the model's confidence in different token predictions evolves through the layers.
            Each line represents one of the top 5 predicted tokens, showing how its probability changes as the input flows through the model.
            """)
            flow_fig = analyzer.visualize_prediction_flow(text, results['tokens'])
            st.plotly_chart(flow_fig, key="flow_plot")
            
            # Attention visualization
            st.subheader("Attention Visualization")
            st.markdown("""
            This heatmap shows how each token in the sequence attends to other tokens, revealing the model's focus patterns at different layers.
            """)
            
            num_layers = len(results['attention'])
            num_heads = results['attention'][0].size(1)
            
            col1, col2 = st.columns(2)
            with col1:
                layer_idx = st.slider("Select transformer layer:", 1, num_layers, 1, key='layer_slider') - 1
            with col2:
                head_idx = st.slider("Select attention head:", 0, num_heads-1, 0, key='head_slider')
            
            # Display attention visualization
            try:
                fig = analyzer.visualize_attention(
                    results['attention'],
                    results['tokens'],
                    layer_idx,
                    head_idx
                )
                st.plotly_chart(fig, key="attention_plot")
                
                st.markdown("""
                ### How to Interpret the Attention Visualization:
                - The heatmap shows how each word (y-axis) attends to other words (x-axis)
                - Brighter colors indicate stronger attention
                - Different layers and heads capture different types of relationships
                - Common Patterns:
                  - Vertical line on first word: Many words attend to the sentence start for context
                  - Diagonal lines: Words paying attention to nearby words
                  - Vertical stripes: Words that are important globally
                  - Bright spots: Related words or grammatically linked words
                """)
                
                # Token Influence visualization
                st.subheader("Token Influence Analysis")
                st.markdown("""
                This visualization shows how much each input token contributes to the model's final prediction, with higher bars indicating stronger influence.
                """)
                
                influence_fig = analyzer.visualize_token_influence(text, results['tokens'])
                st.plotly_chart(influence_fig, key="influence_plot")
                
                # Add Layer Activations Visualization
                st.subheader("Layer Activation Analysis")
                st.markdown("""
                This heatmap shows how strongly each layer responds to different tokens.
                Brighter colors indicate stronger activations, revealing how information flows through the model's layers.
                Early layers often capture basic features while deeper layers develop more abstract representations.
                """)
                activation_fig = analyzer.visualize_layer_activations(text, results['tokens'])
                st.plotly_chart(activation_fig, key="activation_plot")
                
                st.markdown("""
                ### Understanding Layer Activation Patterns:
                - **Embedding Layer (Layer 0)**: Converts tokens into initial vector representations
                - **Early Layers (1-4)**: Process basic features like syntax and word relationships
                - **Middle Layers (5-8)**: Develop more complex patterns and semantic understanding
                - **Late Layers (9-11)**: Show high activation as they assemble final representations
                - **Output Layer (12)**: Shows uniform, lower activation as it normalizes the final layer's output
                
                The bright activations in Layer 11 followed by uniform lower activations in Layer 12 is a typical pattern:
                Layer 11 is making the final computational decisions, while Layer 12 standardizes these outputs for the final prediction.
                """)
                
                # Add AI Safety Implications section
                st.markdown("""
                ### Implications for AI Safety & Interpretability
                
                These visualizations provide important insights for AI safety and interpretability:
                
                - **Model Transparency**: Helps identify potential biases, failure modes, and unexpected dependencies in the model's decision-making process
                
                - **Safety Monitoring**: Can detect unusual activation patterns that might indicate adversarial inputs or model confusion
                
                - **Alignment Verification**: Shows whether the model is attending to appropriate context and basing decisions on relevant information
                
                While these visualizations offer valuable insights into model behavior, they represent only a partial view of the model's internal processing. They are most useful as part of a broader approach to AI interpretability and safety monitoring.
                """)
                
            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please try refreshing the page or contact support if the error persists.")

if __name__ == "__main__":
    main() 