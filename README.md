# Embedding Model Comparison Framework

A comprehensive framework for evaluating and comparing text embedding models. This tool provides metrics for both ground-truth similarity evaluation and document-based evaluation without reference scores.

[![GitHub](https://img.shields.io/badge/GitHub-pierretd/embedding--model--comparison-blue?logo=github)](https://github.com/pierretd/embedding-model-comparison)

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
- **Real-time Progress Tracking**: Monitor evaluation progress with status updates

## Installation

1. Clone this repository:
```bash
git clone https://github.com/pierretd/embedding-model-comparison.git
cd embedding-model-comparison
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

## Evaluation Metrics Explained

### Model Comparison
The framework evaluates embedding models using a set of text pairs with human similarity judgments. It computes:
- **Pearson Correlation**: Measures linear correlation with human judgment scores
- **Spearman Correlation**: Measures rank-order correlation with human judgment scores
- **Embedding Generation Speed**: Time required to generate embeddings (seconds per text)
- **Model Load Time**: Time required to load the model into memory
- **Memory Usage**: Memory footprint of the model during operation

### Document Evaluation
For document evaluation without ground truth data, the framework uses three key metrics:

1. **Paragraph Coherence** (30% of overall score): Measures how semantically similar adjacent paragraphs are compared to non-adjacent ones. Higher values indicate the model better captures the flow of ideas through the document.

   ```
   Example: gte-large (0.8541) > bge-small (0.6947) > all-MiniLM (0.4499)
   ```

2. **Section Boundary Contrast** (30% of overall score): Compares similarity within sections vs. across sections. Higher values show the model effectively distinguishes section boundaries.

   ```
   Example: all-MiniLM (0.2659) > bge-small (0.1073) > gte-large (0.0577)
   ```

3. **Semantic Search Precision** (40% of overall score): Tests the model's ability to retrieve relevant content based on natural language queries. The system automatically generates document-specific queries.

   ```
   Example: gte-large (0.5500) = bge-small (0.5500) > all-MiniLM (0.4333)
   ```

**Overall Score**: Weighted average of the three metrics above. Higher scores indicate better overall performance for document understanding tasks.

## Performance Observations

Based on extensive testing with various documents:

- **GTE-Large**: Excels at paragraph coherence and semantic search, making it ideal for tasks requiring deep semantic understanding. However, it's significantly slower than other models.
- **BGE-Small**: Offers a good balance between performance and speed, with strong results across all metrics.
- **MiniLM**: Fastest model with good section boundary detection, but weaker on paragraph coherence.

## Adding New Models

To add new models for comparison, edit the `models_to_compare` list in both Python files:

```python
models_to_compare = [
    {"name": "model-name", "model_id": "model/identifier"},
    # Add more models here
]
```

## Known Issues

- **FastEmbed Warning**: When using gte-large, you may see a warning about mean pooling vs. CLS embedding. This doesn't impact results but indicates a change in the model's behavior.
- **Multiprocessing on macOS**: On some macOS systems, parallel processing may cause pickling errors. The application will automatically fall back to sequential processing if this occurs.
- **Port Already in Use**: If port 8000 is already in use, specify a different port with `python serve.py --port 8001`.

## Requirements

- Python 3.6+
- fastembed
- numpy, scipy
- tqdm
- other dependencies in requirements.txt

## License

MIT License 