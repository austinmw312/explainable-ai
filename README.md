# GPT-2 Model Analyzer

An interactive web application for visualizing and analyzing how GPT-2 processes and understands text. This tool provides insights into the model's attention patterns and next-token predictions.

## Features

- **Attention Visualization**: See how different parts of the model attend to words in the input text
- **Token Prediction**: View the top 5 most likely next words with their probabilities
- **Interactive Controls**: Explore different layers and attention heads of the model
- **Token Analysis**: Examine how the model tokenizes input text

## Live Demo

Visit [Streamlit Cloud URL] to try the application.

## Running Locally

### Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment (optional but recommended):
```bash
# Using conda
conda create -n model-analyzer python=3.9
conda activate model-analyzer

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

1. Start the Streamlit server:
```bash
streamlit run model_analyzer.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. Enter text in the input area (max 500 characters)
2. Click "Analyze" to process the text
3. Explore the results:
   - View token predictions
   - Use the sliders to explore different attention layers and heads
   - Examine the attention visualization heatmap

## Technical Details

- Uses the base GPT-2 model for analysis
- Runs on CPU for compatibility
- Visualizes attention patterns using Plotly
- Built with Streamlit for the user interface

## Limitations

- Uses the base GPT-2 model only
- Limited to CPU processing
- Maximum input length of 500 characters
- May take a moment to process longer texts
