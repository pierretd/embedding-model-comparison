# Embedding Model Comparison Framework

A comprehensive framework for evaluating and comparing text embedding models. This tool provides metrics for both ground-truth similarity evaluation and document-based evaluation without reference scores.

## Features

### Model Comparison Dashboard
- **Parallel Evaluation**: Evaluates multiple embedding models simultaneously using multiprocessing
- **Performance Metrics**: Measures both correlation with human judgments and computational efficiency
- **Interactive Dashboard**: Visualizes comparison results through interactive charts and tables

### Document Evaluation Dashboard
- **Document Structure Analysis**: Analyzes how well models capture document semantics and structure
- **Multiple Evaluation Metrics**:
  - **Paragraph Coherence**: Measures semantic flow between adjacent paragraphs
  - **Section Boundary Contrast**: Tests models' ability to distinguish content in different vs. same sections
  - **Semantic Search Precision**: Evaluates retrieval accuracy for natural language queries
- **Upload Your Own**: Easily evaluate any text document with the web interface

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/embedding-comparison-framework.git
cd embedding-comparison-framework
```

2. Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

Start the web server to access both dashboards:

```bash
python serve.py
```

This will open your browser to http://localhost:8000 where you can access:
- **Model Comparison**: Compare embedding models against human judgment data
- **Document Evaluation**: Upload and analyze documents with different embedding models

### Running Embedding Comparisons Directly

To run the embedding comparison without the web interface:

```bash
python local_embedding_comparison.py
```

This will evaluate all models in parallel and generate `embedding_comparison_results.json`.

### Running Document Evaluation Directly

To evaluate a document without the web interface:

```bash
python simple_document_evaluation.py path/to/your/document.txt
```

## How It Works

### Model Comparison
The framework evaluates embedding models using a set of text pairs with human similarity judgments. It computes:
- Pearson and Spearman correlations with human judgments
- Embedding generation speed
- Model load time
- Memory usage

### Document Evaluation
For document evaluation without ground truth data, the framework uses three key metrics:

1. **Paragraph Coherence**: Measures how semantically similar adjacent paragraphs are compared to non-adjacent ones. Higher values indicate the model better captures the flow of ideas through the document.

2. **Section Boundary Contrast**: Compares similarity within sections vs. across sections. Higher values show the model effectively distinguishes section boundaries.

3. **Semantic Search Precision**: Tests the model's ability to retrieve relevant content based on natural language queries. The system automatically generates document-specific queries.

## Adding New Models

To add new models for comparison, edit the `models_to_compare` list in both Python files:

```python
models_to_compare = [
    {"name": "model-name", "model_id": "model/identifier"},
    # Add more models here
]
```

## Requirements

- Python 3.6+
- fastembed
- numpy, scipy
- tqdm
- other dependencies in requirements.txt

## License

MIT License 