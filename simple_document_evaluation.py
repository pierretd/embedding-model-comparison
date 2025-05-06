import numpy as np
import re
import json
import time
import os
from scipy.spatial.distance import cosine
from fastembed import TextEmbedding
import multiprocessing
from functools import partial
import threading
from queue import Empty
from tqdm import tqdm
import argparse

# Models to evaluate
MODELS = [
    {"name": "bge-small-en-v1.5", "model_id": "BAAI/bge-small-en-v1.5"},
    {"name": "all-MiniLM-L6-v2", "model_id": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "gte-large", "model_id": "thenlper/gte-large"},
]

# Progress reporter function - will be overridden by the server
def status_reporter(message, progress):
    """Report status of the evaluation process"""
    print(f"Progress: {progress}% - {message}")

# Function to read and pre-process the document
def load_document(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        document = f.read()
    
    # Split into sections and paragraphs
    sections = []
    paragraphs = []
    current_section = ""
    
    # Simple parsing based on blank lines
    for line in document.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Assume shorter lines without periods are section headers
        if len(line) < 50 and not line.endswith('.'):
            if current_section:
                sections.append(current_section)
            current_section = line
        else:
            if current_section:
                # Add to current section
                current_section += "\n" + line
                # Also track as a separate paragraph
                paragraphs.append(line)
    
    # Add the last section
    if current_section:
        sections.append(current_section)
    
    # Split into sentences (simple rule-based approach)
    sentences = []
    for paragraph in paragraphs:
        # Basic sentence splitting by periods, exclamations, and questions
        for sentence in re.split(r'[.!?]', paragraph):
            sentence = sentence.strip()
            if sentence and len(sentence) > 20:  # Only add substantial sentences
                sentences.append(sentence + '.')  # Add back period for completeness
    
    return document, sections, paragraphs, sentences

def load_model(model_info):
    """Load a FastEmbed model."""
    status_reporter(f"Loading model: {model_info['name']}", 30)
    print(f"Loading model: {model_info['name']} ({model_info['model_id']})")
    model = TextEmbedding(model_name=model_info["model_id"])
    return model

def generate_embedding(model, text):
    """Generate embeddings for the text."""
    embedding = list(model.embed([text]))[0]
    return embedding

# 1. Internal Coherence Evaluation
def evaluate_paragraph_coherence(paragraphs, model):
    status_reporter(f"Evaluating paragraph coherence for {len(paragraphs)} paragraphs", 40)
    print(f"  Generating embeddings for {len(paragraphs)} paragraphs...")
    
    # Generate embeddings for all paragraphs with progress bar
    embeddings = []
    for para in tqdm(paragraphs, desc="  Paragraph embeddings", ncols=80):
        embeddings.append(generate_embedding(model, para))
    
    # Calculate coherence between adjacent paragraphs
    coherence_scores = []
    for i in tqdm(range(len(paragraphs)-1), desc="  Computing coherence", ncols=80):
        similarity = 1 - cosine(embeddings[i], embeddings[i+1])
        coherence_scores.append(similarity)
    
    avg_coherence = np.mean(coherence_scores)
    min_coherence = np.min(coherence_scores)
    max_coherence = np.max(coherence_scores)
    
    return {
        "avg_coherence": float(avg_coherence),
        "min_coherence": float(min_coherence),
        "max_coherence": float(max_coherence),
        "coherence_std": float(np.std(coherence_scores))
    }

# 2. Section Similarity vs Cross-Section Similarity
def evaluate_section_boundaries(sections, model):
    status_reporter(f"Evaluating section boundaries for {len(sections)} sections", 60)
    print(f"  Evaluating section boundaries for {len(sections)} sections...")
    
    # Split sections into paragraphs
    section_paragraphs = []
    for section in sections:
        paras = [p for p in section.split('\n') if p and len(p) > 50]
        if paras:
            section_paragraphs.append(paras)
    
    # Compute embeddings for all paragraphs
    all_para_embeddings = []
    section_indices = []  # Keep track of which section each paragraph belongs to
    
    # Show progress for each section
    for i, paras in enumerate(tqdm(section_paragraphs, desc="  Processing sections", ncols=80)):
        embeddings = []
        for para in paras:
            embeddings.append(generate_embedding(model, para))
        all_para_embeddings.extend(embeddings)
        section_indices.extend([i] * len(paras))
    
    # Calculate:
    # 1. Within-section similarity (paragraphs from same section)
    # 2. Cross-section similarity (paragraphs from different sections)
    within_similarities = []
    cross_similarities = []
    
    # Use a tqdm progress bar based on expected comparisons
    total_comparisons = len(all_para_embeddings) * (len(all_para_embeddings) - 1) // 2
    with tqdm(total=total_comparisons, desc="  Computing similarities", ncols=80) as pbar:
        for i in range(len(all_para_embeddings)):
            for j in range(i+1, len(all_para_embeddings)):
                similarity = 1 - cosine(all_para_embeddings[i], all_para_embeddings[j])
                
                # Determine if within same section or across sections
                if section_indices[i] == section_indices[j]:
                    within_similarities.append(similarity)
                else:
                    cross_similarities.append(similarity)
                pbar.update(1)
    
    # Calculate statistics
    avg_within = np.mean(within_similarities) if within_similarities else 0
    avg_cross = np.mean(cross_similarities) if cross_similarities else 0
    
    # The contrast between within vs cross should be higher for better embeddings
    section_contrast = avg_within - avg_cross
    
    return {
        "avg_within_section_similarity": float(avg_within),
        "avg_cross_section_similarity": float(avg_cross),
        "section_boundary_contrast": float(section_contrast),
        "within_similarity_std": float(np.std(within_similarities)) if within_similarities else 0,
        "cross_similarity_std": float(np.std(cross_similarities)) if cross_similarities else 0
    }

# 3. Semantic Search Precision
def evaluate_semantic_search(sentences, model, custom_queries=None):
    status_reporter(f"Evaluating semantic search precision with {len(sentences)} sentences", 80)
    print(f"  Evaluating semantic search precision with {len(sentences)} sentences...")
    
    # Create an index of sentences and their embeddings
    sentence_embeddings = []
    for sentence in tqdm(sentences, desc="  Embedding sentences", ncols=80):
        sentence_embeddings.append(generate_embedding(model, sentence))
    
    # Create test queries that should match specific content
    # If no custom queries, use default ones related to common topics
    if custom_queries is None:
        # Try to detect document topics from keywords
        doc_text = " ".join(sentences).lower()
        
        # Default general queries
        test_queries = [
            "What is the main topic of this document?",
            "What are the key components or features described?",
            "What is the history or background information provided?",
            "What technical details are mentioned?",
            "What benefits or advantages are described?",
            "What challenges or problems are discussed?",
            "What future developments are mentioned?"
        ]
        
        # Expected keywords (will depend on document content)
        expected_keywords = [
            ["main", "topic", "about", "focuses", "primary"],
            ["components", "features", "elements", "parts", "aspects"],
            ["history", "background", "origins", "began", "started"],
            ["technical", "details", "specifications", "numbers", "statistics"],
            ["benefits", "advantages", "positive", "improve", "better"],
            ["challenges", "problems", "issues", "difficulties", "concerns"],
            ["future", "developments", "upcoming", "next", "plans"]
        ]
    else:
        test_queries = custom_queries["queries"]
        expected_keywords = custom_queries["keywords"]
    
    # For each query, find the top 3 most similar sentences
    search_scores = []
    
    for query, keywords in tqdm(zip(test_queries, expected_keywords), desc="  Processing queries", ncols=80, total=len(test_queries)):
        query_embedding = generate_embedding(model, query)
        
        # Calculate similarity to all sentences
        similarities = [1 - cosine(query_embedding, emb) for emb in sentence_embeddings]
        
        # Get top 3 results
        top_indices = np.argsort(similarities)[-3:][::-1]
        top_sentences = [sentences[i] for i in top_indices]
        
        # Check if any expected keywords appear in the results
        keyword_matches = 0
        for sentence in top_sentences:
            sentence_lower = sentence.lower()
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    keyword_matches += 1
                    break  # Count only one match per sentence per keyword
        
        # Score is the proportion of keywords found in top results (max 1.0)
        precision = min(1.0, keyword_matches / len(keywords))
        search_scores.append(precision)
    
    # Calculate average precision
    avg_precision = np.mean(search_scores)
    
    return {
        "semantic_search_precision": float(avg_precision),
        "min_precision": float(np.min(search_scores)),
        "max_precision": float(np.max(search_scores)),
        "search_precision_std": float(np.std(search_scores))
    }

# Generate search queries based on document content
def generate_search_queries(document_text, sections):
    """Generate document-specific search queries based on content analysis"""
    # Simple keyword extraction
    document_lower = document_text.lower()
    search_queries = []
    search_keywords = []
    
    # Try to identify main topics from section headers
    topics = []
    for section in sections:
        lines = section.split('\n')
        if lines and len(lines[0]) < 50:  # First line is likely a header
            topics.append(lines[0])
    
    # Generate queries based on identified topics
    if topics:
        for topic in topics[:min(5, len(topics))]:
            # Create a query about this topic
            query = f"What does the document say about {topic.lower()}?"
            # Extract potential keywords from the topic
            words = re.findall(r'\b\w+\b', topic.lower())
            keywords = [w for w in words if len(w) > 3 and w not in ('what', 'when', 'where', 'which', 'about', 'does')]
            
            if keywords:
                search_queries.append(query)
                search_keywords.append(keywords + ["information", "details"])
    
    # If we couldn't extract enough topic-based queries, use generic ones
    if len(search_queries) < 3:
        search_queries = [
            "What is the main subject of this document?",
            "What key information is presented in this text?", 
            "What are the main points discussed?",
            "What conclusions or recommendations are made?",
            "What evidence or examples are provided?"
        ]
        search_keywords = [
            ["main", "subject", "topic", "about", "focuses"],
            ["key", "information", "important", "central", "critical"],
            ["points", "arguments", "ideas", "concepts", "discussed"],
            ["conclusions", "recommendations", "summary", "suggests", "proposes"],
            ["evidence", "examples", "instances", "cases", "illustrations"]
        ]
    
    return {
        "queries": search_queries[:7],  # Limit to 7 queries
        "keywords": search_keywords[:7]
    }

def evaluate_single_model(model_info, document_data, search_queries):
    """Evaluate a single model on the document"""
    model_name = model_info["name"]
    
    # Unpack document data
    document, sections, paragraphs, sentences = document_data
    
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_name}")
    
    try:
        start_time = time.time()
        
        # Load model
        model = load_model(model_info)
        
        # Run evaluations
        print("1. Evaluating paragraph coherence...")
        coherence_result = evaluate_paragraph_coherence(paragraphs, model)
        
        print("2. Evaluating section boundaries...")
        section_result = evaluate_section_boundaries(sections, model)
        
        print("3. Evaluating semantic search precision...")
        search_result = evaluate_semantic_search(sentences, model, search_queries)
        
        # Calculate overall score (weighted average of key metrics)
        overall_score = (
            coherence_result["avg_coherence"] * 0.3 +
            section_result["section_boundary_contrast"] * 0.3 +
            search_result["semantic_search_precision"] * 0.4
        )
        
        model_result = {
            "model_id": model_info["model_id"],
            "coherence": coherence_result,
            "section_boundaries": section_result,
            "semantic_search": search_result,
            "evaluation_time_seconds": time.time() - start_time,
            "overall_score": float(overall_score)
        }
        
        print(f"Evaluation complete: {model_name}")
        print(f"- Average paragraph coherence: {coherence_result['avg_coherence']:.4f}")
        print(f"- Section boundary contrast: {section_result['section_boundary_contrast']:.4f}")
        print(f"- Semantic search precision: {search_result['semantic_search_precision']:.4f}")
        print(f"- Overall score: {overall_score:.4f}")
        
        return {model_name: model_result}
        
    except Exception as e:
        print(f"Error evaluating model {model_name}: {str(e)}")
        return None

# Main evaluation function
def evaluate_document_embeddings(document_path, models, output_file="document_evaluation_results.json"):
    global status_reporter
    
    try:
        status_reporter("Loading document", 15)
        print(f"Loading document: {document_path}")
        document_data = load_document(document_path)
        document, sections, paragraphs, sentences = document_data
        
        print(f"Document statistics:")
        print(f"- Number of sections: {len(sections)}")
        print(f"- Number of paragraphs: {len(paragraphs)}")
        print(f"- Number of sentences: {len(sentences)}")
        
        # Generate search queries based on document content
        print("Generating document-specific search queries...")
        search_queries = generate_search_queries(document, sections)
        print(f"Generated {len(search_queries['queries'])} queries based on document content")
        
        # Initialize start time
        start_time = time.time()
        
        results = {}
        
        # Process models sequentially
        for i, model_info in enumerate(models):
            progress_base = 20 + (i / len(models)) * 70
            status_reporter(f"Evaluating model {i+1}/{len(models)}: {model_info['name']}", int(progress_base))
            
            # Evaluate the model
            model_result = evaluate_single_model(model_info, document_data, search_queries)
            
            # Add to results if successful
            if model_result:
                results.update(model_result)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        
        # Save results
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        status_reporter("Evaluation complete, saving results", 95)
        print(f"\nResults saved to {output_file}")
        return results
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    # To avoid issues with multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate embedding models on documents.')
    parser.add_argument('document', type=str, help='Path to the document text file to evaluate')
    parser.add_argument('--models', type=str, nargs='+', choices=[m["name"] for m in MODELS], 
                        help='Specific models to evaluate (default: all)')
    parser.add_argument('--output', type=str, default="document_evaluation_results.json",
                        help='Output file path (default: document_evaluation_results.json)')
    
    args = parser.parse_args()
    
    # Validate document path
    if not os.path.exists(args.document):
        print(f"Error: Document not found at {args.document}")
        exit(1)
    
    # Select models to evaluate
    if args.models:
        selected_models = [m for m in MODELS if m["name"] in args.models]
        if not selected_models:
            print(f"Error: No valid models selected. Available models: {[m['name'] for m in MODELS]}")
            exit(1)
    else:
        selected_models = MODELS
    
    print(f"Evaluating {len(selected_models)} models on document: {args.document}")
    evaluate_document_embeddings(args.document, selected_models, args.output)
else:
    # Make sure to set spawn method when imported too
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, which is fine
        pass 