import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import time
import os
import multiprocessing
from functools import partial

# Import embedding models
from fastembed import TextEmbedding

# Evaluation dataset - sample pairs with human scores
evaluation_pairs = [
    {
        "text1": "I love listening to rock music.",
        "text2": "Rock music is my favorite genre.",
        "human_score": 0.9
    },
    {
        "text1": "This album has complex jazz fusion elements.",
        "text2": "The record incorporates experimental jazz techniques.",
        "human_score": 0.85
    },
    {
        "text1": "The vocalist has a powerful voice.",
        "text2": "The guitar solos are impressive.",
        "human_score": 0.3
    },
    {
        "text1": "This hip-hop album addresses social issues.",
        "text2": "The rapper discusses political topics in their lyrics.",
        "human_score": 0.8
    },
    {
        "text1": "The production quality is excellent.",
        "text2": "The album sounds terrible.",
        "human_score": 0.1
    }
]

# Models to compare - updated with supported models
models_to_compare = [
    {"name": "bge-small-en-v1.5", "model_id": "BAAI/bge-small-en-v1.5"},
    {"name": "bge-base-en-v1.5", "model_id": "BAAI/bge-base-en-v1.5"},
    {"name": "all-MiniLM-L6-v2", "model_id": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "multilingual-e5-large", "model_id": "intfloat/multilingual-e5-large"},
    {"name": "multilingual-MiniLM-L12-v2", "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"},
    {"name": "gte-large", "model_id": "thenlper/gte-large"},
    {"name": "nomic-embed-text-v1.5", "model_id": "nomic-ai/nomic-embed-text-v1.5"}
]

def load_model(model_info):
    """Load a FastEmbed model."""
    print(f"Loading model: {model_info['name']} ({model_info['model_id']})")
    model = TextEmbedding(model_name=model_info["model_id"])
    return model

def generate_embeddings(model, text):
    """Generate embeddings for the text."""
    # FastEmbed returns a generator, convert to a list and get the first item
    embedding = list(model.embed([text]))[0]
    return embedding

def evaluate_model(model_info):
    """Evaluate a single embedding model on the evaluation pairs."""
    process_name = multiprocessing.current_process().name
    print(f"\n{process_name} - {'='*60}")
    print(f"{process_name} - Evaluating model: {model_info['name']} ({model_info['model_id']})")
    print(f"{process_name} - {'='*60}")
    
    results = []
    
    try:
        # Load the model
        start_time = time.time()
        model = load_model(model_info)
        load_time = time.time() - start_time
        print(f"{process_name} - Model loaded in {load_time:.2f} seconds")
        
        # Process each pair
        embedding_start_time = time.time()
        for i, pair in enumerate(evaluation_pairs):
            print(f"{process_name} - Processing pair {i+1}/{len(evaluation_pairs)}: {pair['text1'][:30]}...")
            
            # Generate embeddings
            embedding1 = generate_embeddings(model, pair["text1"])
            embedding2 = generate_embeddings(model, pair["text2"])
            
            # Calculate cosine similarity (1 - cosine distance)
            similarity = 1 - cosine(embedding1, embedding2)
            
            # Store results
            result = {
                "text1": pair["text1"],
                "text2": pair["text2"],
                "human_score": pair["human_score"],
                "similarity": float(similarity),
                "embedding1_sample": embedding1[:5].tolist(),
                "embedding2_sample": embedding2[:5].tolist(),
                "embedding_dimension": len(embedding1)
            }
            results.append(result)
            print(f"{process_name} - Similarity: {similarity:.4f}, Human score: {pair['human_score']}")
        
        embedding_time = time.time() - embedding_start_time
        avg_embed_time = embedding_time / (len(evaluation_pairs) * 2)  # Two embeddings per pair
        
        # Calculate correlations
        human_scores = [r["human_score"] for r in results]
        model_scores = [r["similarity"] for r in results]
        pearson_corr, p_value = pearsonr(human_scores, model_scores)
        spearman_corr, s_p_value = spearmanr(human_scores, model_scores)
        
        model_result = {
            "model_name": model_info["name"],
            "model_id": model_info["model_id"],
            "load_time_seconds": load_time,
            "average_embedding_time_seconds": avg_embed_time,
            "pearson_correlation": float(pearson_corr),
            "spearman_correlation": float(spearman_corr),
            "embedding_dimension": results[0]["embedding_dimension"],
            "pairs": results
        }
        
        print(f"{process_name} - Model evaluation complete:")
        print(f"{process_name} -   Pearson correlation: {pearson_corr:.4f}")
        print(f"{process_name} -   Spearman correlation: {spearman_corr:.4f}")
        print(f"{process_name} -   Embedding dimension: {results[0]['embedding_dimension']}")
        print(f"{process_name} -   Load time: {load_time:.2f} seconds")
        print(f"{process_name} -   Average embedding time: {avg_embed_time:.4f} seconds per text")
        
        return model_result
    
    except Exception as e:
        print(f"{process_name} - Error evaluating model {model_info['name']}: {str(e)}")
        return None

def run_parallel_comparison():
    """Run the comparison of all models in parallel."""
    print(f"Starting parallel FastEmbed model comparison with {len(models_to_compare)} models")
    
    # Determine optimal number of processes
    num_models = len(models_to_compare)
    cpu_count = multiprocessing.cpu_count()
    num_processes = min(num_models, cpu_count)
    
    print(f"Using {num_processes} processes for {num_models} models (system has {cpu_count} CPUs)")
    
    # Initialize start time
    start_time = time.time()
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the evaluate_model function to all models
        results_list = pool.map(evaluate_model, models_to_compare)
    
    # Filter out None results (failed evaluations)
    valid_results = [r for r in results_list if r is not None]
    
    # Create a dictionary from the results
    results = {r["model_name"]: r for r in valid_results}
    successful_models = [r["model_name"] for r in valid_results]
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results to file
    if results:
        output_file = "embedding_comparison_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
        # Print comparison table
        print("\n" + "="*110)
        print("MODEL COMPARISON SUMMARY")
        print("="*110)
        print(f"{'Model Name':<30} {'Pearson':<10} {'Spearman':<10} {'Dimensions':<10} {'Load Time (s)':<15} {'Embed Time (s)':<15}")
        print("-"*110)
        
        for model_name in successful_models:
            model_data = results[model_name]
            print(f"{model_name:<30} {model_data['pearson_correlation']:<10.4f} {model_data['spearman_correlation']:<10.4f} {model_data['embedding_dimension']:<10} {model_data['load_time_seconds']:<15.2f} {model_data['average_embedding_time_seconds']:<15.4f}")
        
        print("-"*110)
        print(f"Total execution time: {total_time:.2f} seconds with parallel processing ({num_processes} processes)")
        print("="*110)
    else:
        print("No successful model evaluations to report.")

if __name__ == "__main__":
    # To avoid issues with multiprocessing on macOS
    multiprocessing.set_start_method('spawn')
    run_parallel_comparison() 