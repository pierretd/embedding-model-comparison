import http.server
import socketserver
import webbrowser
import os
import threading
import time
import json
import cgi
import sys
import traceback
import tempfile
import importlib.util
import multiprocessing
import argparse
from urllib.parse import parse_qs, urlparse

# Default port for the web server
PORT = 8000

# Global variables for tracking evaluation status
evaluation_status = {
    "status": "idle",  # idle, processing, completed, error
    "message": "",
    "progress": 0,
    "document_name": ""
}

# Path to store evaluation history
EVALUATION_HISTORY_FILE = "evaluation_history.json"

# Output file for document evaluations
DOCUMENT_EVAL_RESULTS = "document_evaluation_results.json"
EMBEDDING_RESULTS = "embedding_comparison_results.json"

# Load any previous document name if available
try:
    if os.path.exists(EVALUATION_HISTORY_FILE):
        with open(EVALUATION_HISTORY_FILE, 'r') as f:
            history = json.load(f)
            if 'last_document' in history:
                evaluation_status['document_name'] = history['last_document']
except Exception as e:
    print(f"Error loading evaluation history: {str(e)}")

# Save the current document name for persistence
def save_evaluation_history():
    try:
        history = {}
        if os.path.exists(EVALUATION_HISTORY_FILE):
            with open(EVALUATION_HISTORY_FILE, 'r') as f:
                try:
                    history = json.load(f)
                except:
                    history = {}
        
        history['last_document'] = evaluation_status['document_name']
        history['last_evaluation_time'] = time.time()
        
        with open(EVALUATION_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        print(f"Error saving evaluation history: {str(e)}")

# Set multiprocessing start method to 'spawn' for macOS compatibility
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Method already set, which is fine
    pass

# Custom HTTP request handler
class CustomHandler(http.server.SimpleHTTPRequestHandler):
    # Set default headers for all responses
    def end_headers(self):
        # Add CORS headers to allow fetch from any origin
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight requests"""
        self.send_response(200)
        self.end_headers()
    
    def log_message(self, format, *args):
        """Override to show cleaner logs"""
        print(f"[{self.log_date_time_string()}] {args[0]}")
    
    def do_GET(self):
        """Handle GET requests"""
        # Special endpoint for evaluation status
        if self.path == '/evaluation_status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(evaluation_status).encode())
            return
        
        # Default handler for static files
        super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        global evaluation_status
        
        if self.path == '/upload_document':
            try:
                # Check if an evaluation is already running
                if evaluation_status["status"] == "processing":
                    self.send_response(409)  # Conflict
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "error": "An evaluation is already in progress.",
                        "current_document": evaluation_status["document_name"]
                    }).encode())
                    return
                
                # Parse the form data
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST',
                             'CONTENT_TYPE': self.headers['Content-Type']}
                )
                
                # Check if the file field is present
                if 'document' not in form:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "No document file provided."}).encode())
                    return
                
                # Get the file field
                fileitem = form['document']
                
                # Check if it's a file
                if not fileitem.file:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Not a valid file."}).encode())
                    return
                
                # Get the filename and content
                filename = fileitem.filename
                if not filename.endswith('.txt'):
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Only text (.txt) files are supported."}).encode())
                    return
                
                # Get use_default_models parameter
                use_default_models = True
                if 'use_default_models' in form:
                    use_default_models = form.getvalue('use_default_models').lower() in ('true', 'yes', '1', 'on')
                
                # Write the file to a temporary location
                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, filename)
                
                with open(file_path, 'wb') as f:
                    f.write(fileitem.file.read())
                
                # Update status
                evaluation_status = {
                    "status": "processing",
                    "message": "Document received. Starting evaluation...",
                    "progress": 10,
                    "document_name": filename
                }
                
                # Save document info for persistence
                save_evaluation_history()
                
                # Start evaluation in a separate thread
                eval_thread = threading.Thread(
                    target=run_document_evaluation,
                    args=(file_path, use_default_models, DOCUMENT_EVAL_RESULTS)
                )
                eval_thread.daemon = True
                eval_thread.start()
                
                # Send successful response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": "Document uploaded successfully. Evaluation started.",
                    "filename": filename
                }).encode())
                
            except Exception as e:
                # Handle any errors
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }).encode())
                
                # Update status
                evaluation_status = {
                    "status": "error",
                    "message": f"Error processing upload: {str(e)}",
                    "progress": 0,
                    "document_name": ""
                }
        else:
            # Handle unknown POST endpoints
            self.send_response(404)
            self.end_headers()

def run_document_evaluation(document_path, use_default_models=True, output_file=DOCUMENT_EVAL_RESULTS):
    """Run the document evaluation script in a background thread"""
    global evaluation_status
    
    try:
        # Update status
        evaluation_status["message"] = "Loading models and document..."
        evaluation_status["progress"] = 20
        
        # Import the evaluation module directly
        try:
            # Import the module
            spec = importlib.util.spec_from_file_location("document_eval", "simple_document_evaluation.py")
            doc_eval_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(doc_eval_module)
            
            # Create a status reporter that updates the global status
            def status_reporter(message, progress):
                global evaluation_status
                evaluation_status["message"] = message
                evaluation_status["progress"] = progress
                print(f"Status update: {progress}% - {message}")
            
            # Set the status reporter in the module
            doc_eval_module.status_reporter = status_reporter
            
            # Use default models or customize
            models = doc_eval_module.MODELS if use_default_models else [
                {"name": "bge-small-en-v1.5", "model_id": "BAAI/bge-small-en-v1.5"}
            ]
            
            print(f"Starting parallel evaluation with {len(models)} models...")
            
            # Run the evaluation
            results = doc_eval_module.evaluate_document_embeddings(document_path, models, output_file)
            
            # Update status
            evaluation_status["status"] = "completed"
            evaluation_status["message"] = "Evaluation completed successfully."
            evaluation_status["progress"] = 100
            
        except Exception as e:
            raise Exception(f"Error importing or running evaluation module: {str(e)}")
        
    except Exception as e:
        # Update status on error
        evaluation_status["status"] = "error"
        evaluation_status["message"] = f"Evaluation failed: {str(e)}"
        evaluation_status["progress"] = 0
        print(f"Error in document evaluation: {str(e)}")
        traceback.print_exc()

def open_browser(url):
    """Function to open web browser after a short delay"""
    time.sleep(1)
    print(f"Opening browser at {url}")
    webbrowser.open(url)

def start_server(port=PORT, open_browser_on_start=True, dashboard="document_eval_dashboard.html"):
    """Start the HTTP server"""
    # Allow reuse of the port
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", port), CustomHandler) as httpd:
            print(f"Server started at http://localhost:{port}")
            print("Press Ctrl+C to stop the server")
            
            # Open browser in a separate thread if requested
            if open_browser_on_start:
                browser_url = f"http://localhost:{port}/{dashboard}"
                threading.Thread(target=open_browser, args=(browser_url,), daemon=True).start()
            
            # Start the server in the main thread
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"Error: Port {port} is already in use.")
            print("Try stopping any running servers or use a different port with --port.")
            sys.exit(1)
        else:
            raise

def check_required_files(required_files):
    """Check if required files exist"""
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Error: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    return True

def create_empty_results_file(file_path):
    """Create an empty results file if it doesn't exist"""
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({"empty": {
                "model_id": "none",
                "coherence": {"avg_coherence": 0, "min_coherence": 0, "max_coherence": 0, "coherence_std": 0},
                "section_boundaries": {
                    "avg_within_section_similarity": 0,
                    "avg_cross_section_similarity": 0,
                    "section_boundary_contrast": 0,
                    "within_similarity_std": 0,
                    "cross_similarity_std": 0
                },
                "semantic_search": {
                    "semantic_search_precision": 0,
                    "min_precision": 0,
                    "max_precision": 0,
                    "search_precision_std": 0
                },
                "evaluation_time_seconds": 0,
                "overall_score": 0
            }}, f, indent=2)
        print(f"Created empty results file: {file_path}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Start the embedding comparison web server.')
    parser.add_argument('--port', type=int, default=PORT, help=f'Port to run the server on (default: {PORT})')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--dashboard', choices=['index.html', 'document_eval_dashboard.html'], 
                      default='document_eval_dashboard.html', help='Which dashboard to open by default')
    
    args = parser.parse_args()
    
    # Check required files
    required_files = ["index.html", "document_eval_dashboard.html", "simple_document_evaluation.py"]
    if not check_required_files(required_files):
        sys.exit(1)
    
    # Create empty results files if they don't exist
    create_empty_results_file(DOCUMENT_EVAL_RESULTS)
    
    if not os.path.exists(EMBEDDING_RESULTS):
        print(f"Warning: {EMBEDDING_RESULTS} not found.")
        print("Run local_embedding_comparison.py to generate embedding comparison results.")
    
    # Start the server
    start_server(port=args.port, open_browser_on_start=not args.no_browser, dashboard=args.dashboard) 