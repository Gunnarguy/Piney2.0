import os
import re
import sys
import time
import argparse
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

from openai import OpenAI
import pinecone
import tiktoken
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into os.environ

# Optional libraries for file processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

# -------------------- CONFIGURATION FUNCTIONS --------------------
def load_config():
    pinecone_key = os.getenv("PINECONE_API_KEY", "your_default_pinecone_key")
    openai_key = os.getenv("OPENAI_API_KEY", "your_default_openai_key")
    pinecone_env = os.getenv("PINECONE_ENV", "us-east1-gcp")
    if not pinecone_key or not openai_key:
        raise EnvironmentError("Please set both PINECONE_API_KEY and OPENAI_API_KEY environment variables.")
    return pinecone_key, openai_key, pinecone_env

# -------------------- TEXT PROCESSING FUNCTIONS --------------------
def clean_text(text):
    artifacts = [r"endobj", r"xref", r"stream", r"endstream", r"trailer", r"%%EOF", r"obj"]
    for pattern in artifacts:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_tokens=1000):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_str = enc.decode(chunk_tokens)
        chunks.append(chunk_str)
        start = end
    return chunks

# -------------------- FILE PROCESSING --------------------
def process_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    try:
        if ext in ['.txt', '.md', '.py', '.json', '.csv']:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == '.pdf' and PyPDF2:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    text += page_text
        elif ext in ['.docx'] and docx:
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    text = clean_text(text)
    return chunk_text(text)

def process_directory(directory):
    documents = []
    doc_id = 1
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            chunks = process_file(filepath)
            for chunk in chunks:
                if chunk.strip():
                    documents.append({
                        "id": f"doc{doc_id}",
                        "text": chunk,
                        "source": filepath
                    })
                    doc_id += 1
    return documents

# -------------------- PINECONE & EMBEDDING FUNCTIONS --------------------
def initialize_clients():
    pinecone_key, openai_key, pinecone_env = load_config()
    openai_client = OpenAI(api_key=openai_key)
    pc = pinecone.Pinecone(api_key=pinecone_key)
    return pc, pinecone_env, openai_client

def select_or_create_index(pc, non_interactive=False, provided_index=None):
    index_data = pc.list_indexes()
    existing_names = [item["name"] for item in index_data.get("indexes", [])]
    
    if provided_index:
        if provided_index in existing_names:
            print(f"Using provided existing index: {provided_index}")
            return provided_index
        else:
            print(f"Provided index '{provided_index}' does not exist. Creating it...")
            pc.create_index(name=provided_index, dimension=3072, metric="cosine")
            wait_for_index_ready(provided_index, pc)
            return provided_index

    index_choice = None
    if existing_names:
        index_choice = simple_input("Existing indexes: " + ", ".join(existing_names) +
                                     "\nEnter the name of the index to use (or leave blank to create new): ")
        if index_choice and index_choice in existing_names:
            return index_choice
    index_choice = simple_input("Enter a name for the new index: ")
    if not index_choice:
        index_choice = "default_index"
    pc.create_index(name=index_choice, dimension=3072, metric="cosine")
    wait_for_index_ready(index_choice, pc)
    return index_choice

def wait_for_index_ready(index_name, pc):
    while True:
        status = pc.describe_index(index_name).status
        if status.get("ready", False):
            print(f"Index '{index_name}' is ready!")
            break
        else:
            print(f"Waiting for index '{index_name}' to become ready...")
            time.sleep(2)

def list_namespaces(index):
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces", {})
    if namespaces:
        return list(namespaces.keys())
    else:
        return []

def embed_text(texts, client):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [data.embedding for data in response.data]

def upsert_data(index, documents, namespace, openai_client, batch_size=100):
    texts = [doc["text"] for doc in documents]
    embeddings = embed_text(texts, openai_client)
    vectors = []
    for doc, emb in zip(documents, embeddings):
        vectors.append({
            "id": doc["id"],
            "values": emb,
            "metadata": {"text": doc["text"], "source": doc["source"]}
        })
    total_batches = (len(vectors) + batch_size - 1) // batch_size
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted batch {i//batch_size + 1} of {total_batches}")
    print(f"Successfully upserted {len(vectors)} documents into index '{index.name}' under namespace '{namespace}'.")

def query_pinecone(index, namespace, query, openai_client, top_k=3):
    query_embedding = embed_text([query], openai_client)[0]
    results = index.query(
        namespace=namespace,
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    return results

# -------------------- SIMPLE UTILS --------------------
def simple_input(prompt_text):
    return input(prompt_text)

# -------------------- UI CLASS USING TKINTER --------------------
class RAGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Pipeline UI")
        self.geometry("800x600")
        self.pc = None
        self.index = None
        self.namespace = None
        self.documents = []
        self.openai_client = None
        
        self.create_widgets()
        try:
            self.pc, self.pinecone_env, self.openai_client = initialize_clients()
            self.log("Pinecone and OpenAI clients initialized.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def create_widgets(self):
        # Frame for index/namespace selection
        self.frame_config = tk.LabelFrame(self, text="Configuration", padx=10, pady=10)
        self.frame_config.pack(fill="x", padx=10, pady=5)
        
        tk.Label(self.frame_config, text="Index:").grid(row=0, column=0, sticky="e")
        self.entry_index = tk.Entry(self.frame_config, width=30)
        self.entry_index.grid(row=0, column=1, padx=5)
        
        tk.Button(self.frame_config, text="Select/Create Index", command=self.handle_index_selection).grid(row=0, column=2, padx=5)
        
        tk.Label(self.frame_config, text="Namespace:").grid(row=1, column=0, sticky="e")
        self.entry_namespace = tk.Entry(self.frame_config, width=30)
        self.entry_namespace.grid(row=1, column=1, padx=5)
        
        tk.Button(self.frame_config, text="List Namespaces", command=self.handle_list_namespaces).grid(row=1, column=2, padx=5)
        
        # Frame for document ingestion
        self.frame_ingest = tk.LabelFrame(self, text="Document Ingestion", padx=10, pady=10)
        self.frame_ingest.pack(fill="x", padx=10, pady=5)
        
        self.label_directory = tk.Label(self.frame_ingest, text="No directory selected.")
        self.label_directory.pack(side="left", padx=5)
        
        tk.Button(self.frame_ingest, text="Select Directory", command=self.select_directory).pack(side="left", padx=5)
        tk.Button(self.frame_ingest, text="Process & Upsert", command=self.process_and_upsert).pack(side="left", padx=5)
        
        # Frame for query interface
        self.frame_query = tk.LabelFrame(self, text="Query", padx=10, pady=10)
        self.frame_query.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.entry_query = tk.Entry(self.frame_query, width=60)
        self.entry_query.pack(side="top", padx=5, pady=5)
        
        tk.Button(self.frame_query, text="Run Query", command=self.run_query).pack(side="top", padx=5, pady=5)
        
        self.text_results = scrolledtext.ScrolledText(self.frame_query, wrap=tk.WORD, height=10)
        self.text_results.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Log output
        self.text_log = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=10)
        self.text_log.pack(fill="both", expand=True, padx=10, pady=5)
    
    def log(self, message):
        self.text_log.insert(tk.END, message + "\n")
        self.text_log.see(tk.END)
    
    def handle_index_selection(self):
        idx = self.entry_index.get().strip()
        self.index = None
        if not idx:
            idx = simple_input("Enter a name for the new index: ")
        self.entry_index.delete(0, tk.END)
        self.entry_index.insert(0, idx)
        try:
            selected_index = select_or_create_index(pc=self.pc, non_interactive=True, provided_index=idx)
            self.index = self.pc.Index(selected_index)
            self.log(f"Using index: {selected_index}")
        except Exception as e:
            messagebox.showerror("Index Selection Error", str(e))
    
    def handle_list_namespaces(self):
        if self.index is None:
            messagebox.showerror("Error", "No index selected!")
            return
        ns_list = list_namespaces(self.index)
        if ns_list:
            messagebox.showinfo("Namespaces", "Existing namespaces:\n" + "\n".join(ns_list))
        else:
            messagebox.showinfo("Namespaces", "No namespaces found in this index.")
    
    def select_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.label_directory.config(text=directory)
            self.documents = process_directory(directory)
            self.log(f"Processed {len(self.documents)} document chunks from {directory}")
        else:
            self.log("No directory selected.")
    
    def process_and_upsert(self):
        if self.index is None:
            messagebox.showerror("Error", "Please select or create an index first.")
            return
        ns = self.entry_namespace.get().strip()
        if not ns:
            ns = "default_namespace"
            self.entry_namespace.insert(0, ns)
        self.namespace = ns
        try:
            upsert_data(self.index, self.documents, self.namespace, self.openai_client)
            self.log("Documents upserted successfully.")
        except Exception as e:
            messagebox.showerror("Upsert Error", str(e))
    
    def run_query(self):
        if self.index is None or not self.namespace:
            messagebox.showerror("Error", "Ensure an index is selected and a namespace is set.")
            return
        query = self.entry_query.get().strip()
        if not query:
            messagebox.showerror("Error", "Please enter a query.")
            return
        try:
            results = query_pinecone(self.index, self.namespace, query, self.openai_client)
            self.text_results.delete("1.0", tk.END)
            self.text_results.insert(tk.END, "Top Results:\n")
            for match in results.get("matches", []):
                result_text = f"- {match['metadata']['text']} (Score: {match['score']:.4f})\n"
                self.text_results.insert(tk.END, result_text)
            self.log("Query executed successfully.")
        except Exception as e:
            messagebox.showerror("Query Error", str(e))

if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()
