import os
import argparse
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken
import sys
from dotenv import load_dotenv

# Optional file processing imports
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx
except ImportError:
    docx = None

load_dotenv()

# -------------------- CONFIGURATION --------------------
def load_config():
    pinecone_key = os.getenv("PINECONE_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    
    if not pinecone_key or not openai_key:
        raise EnvironmentError("Missing Pinecone or OpenAI API keys")
    return pinecone_key, openai_key

# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Pinecone Embedding Pipeline")
    parser.add_argument("--index", help="Pinecone index name")
    parser.add_argument("--namespace", help="Namespace to use")
    parser.add_argument("--directory", help="Document directory path")
    parser.add_argument("--non-interactive", action="store_true", 
                      help="Non-interactive mode")
    return parser.parse_args()

# -------------------- INDEX MANAGEMENT --------------------
def show_index_namespaces(index):
    """Display namespaces without duplication"""
    try:
        stats = index.describe_index_stats()
        if stats.namespaces:
            print("\nExisting namespaces:")
            for ns in stats.namespaces.keys():
                print(f"  - {ns}")
        else:
            print("\nNo existing namespaces")
    except Exception as e:
        print(f"\nNamespace check error: {str(e)}")

def select_or_create_index(pc, args):
    """Index selection with single namespace display"""
    if args.index:
        if args.index in pc.list_indexes().names():
            print(f"Using existing index: {args.index}")
            index = pc.Index(args.index)
            show_index_namespaces(index)
            return args.index, index
        else:
            print(f"Creating new index: {args.index}")
            pc.create_index(
                name=args.index,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            return args.index, pc.Index(args.index)

    if args.non_interactive:
        default_index = "default_index"
        print(f"Using default index: {default_index}")
        return default_index, pc.Index(default_index)

    existing = pc.list_indexes().names()
    print(f"\nExisting indexes: {existing}")
    
    while True:
        choice = input("[C]reate new or [S]elect existing index: ").upper()
        if choice == "S":
            index_name = input("Enter index name: ").strip()
            if index_name in existing:
                index = pc.Index(index_name)
                show_index_namespaces(index)
                return index_name, index
            else:
                print(f"Index {index_name} not found")
        elif choice == "C":
            index_name = input("New index name: ").strip()
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            return index_name, pc.Index(index_name)

# -------------------- NAMESPACE MANAGEMENT --------------------
def select_namespace(index, args):
    """Namespace selection without duplicate listing"""
    try:
        stats = index.describe_index_stats()
        existing_ns = list(stats.namespaces.keys())
    except Exception as e:
        print(f"Namespace error: {str(e)}")
        existing_ns = []

    if args.namespace:
        print(f"\nUsing provided namespace: {args.namespace}")
        return args.namespace

    if args.non_interactive:
        default_ns = "default_ns"
        print(f"\nUsing default namespace: {default_ns}")
        return default_ns

    print("\nNamespace options:")
    print("1. Existing namespace")
    print("2. Create new")
    print("3. Use default")
    
    while True:
        choice = input("Select option [1-3]: ").strip()
        if choice == "1":
            if not existing_ns:
                print("No existing namespaces")
                continue
            return input("Enter namespace name: ").strip()
        elif choice == "2":
            return input("New namespace name: ").strip()
        elif choice == "3":
            return None
        else:
            print("Invalid choice")

# -------------------- DOCUMENT PROCESSING --------------------
def chunk_text(text, max_tokens=1000):
    """Token-aware chunking with progress"""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return [enc.decode(tokens[i:i+max_tokens]) 
            for i in range(0, len(tokens), max_tokens)]

def process_file(filepath):
    """File processing with chunk reporting"""
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    
    try:
        if ext in ['.txt', '.md', '.py', '.json', '.csv']:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext == '.pdf' and PyPDF2:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "\n".join([page.extract_text() or "" 
                                for page in reader.pages])
        elif ext == '.docx' and docx:
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
    
    chunks = chunk_text(text)
    print(f"Processed {filepath} → {len(chunks)} chunks")
    return chunks

def process_directory(directory):
    """Directory processing with detailed reporting"""
    documents = []
    total_files = 0
    total_chunks = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            chunks = process_file(filepath)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    documents.append({
                        "id": f"doc{total_files}_{i}",
                        "text": chunk
                    })
                    total_chunks += 1
            total_files += 1
            
    print(f"\nProcessing Summary:")
    print(f" - Total files: {total_files}")
    print(f" - Total chunks: {total_chunks}")
    print(f" - Avg chunks/file: {total_chunks/total_files:.1f}" 
          if total_files else "N/A")
    
    return documents

# -------------------- CORE OPERATIONS --------------------
def embed_text(client, texts):
    """Batch embedding with OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return [r.embedding for r in response.data]

def upsert_data(index, namespace, documents):
    """Optimized upsert with progress tracking"""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch_size = 100
    total = len(documents)
    
    print(f"\nStarting upsert of {total} vectors...")
    
    for i in range(0, total, batch_size):
        batch = documents[i:i+batch_size]
        texts = [doc["text"] for doc in batch]
        embeddings = embed_text(client, texts)
        
        vectors = [{
            "id": doc["id"],
            "values": emb,
            "metadata": {"text": doc["text"]}
        } for doc, emb in zip(batch, embeddings)]
        
        index.upsert(
            vectors=vectors,
            namespace=namespace
        )
        print(f"Upserted batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")

# -------------------- MAIN EXECUTION --------------------
def main():
    args = parse_args()
    pinecone_key, openai_key = load_config()
    
    pc = Pinecone(api_key=pinecone_key)
    oai = openai.OpenAI(api_key=openai_key)
    
    # Index handling
    index_name, index = select_or_create_index(pc, args)
    
    # Namespace handling
    namespace = select_namespace(index, args)
    print(f"\nSelected namespace: {namespace or 'default'}")
    
    # Document processing
    directory = args.directory or ("./documents" if args.non_interactive 
                                  else input("Document directory: "))
    if not os.path.isdir(directory):
        print(f"Invalid directory: {directory}")
        sys.exit(1)
        
    documents = process_directory(directory)
    if not documents:
        print("No documents processed")
        sys.exit(1)
        
    # Data upsert
    upsert_data(index, namespace, documents)
    
    # Interactive mode
    if not args.non_interactive:
        while True:
            choice = input("\nQuery index? [y/n]: ").lower()
            if choice != "y":
                break
                
            query = input("Search query: ").strip()
            embedding = embed_text(oai, [query])[0]
            results = index.query(
                namespace=namespace,
                vector=embedding,
                top_k=5,
                include_metadata=True
            )
            
            print("\nTop results:")
            for i, match in enumerate(results["matches"]):
                print(f"{i+1}. {match['metadata']['text'][:100]}... (Score: {match['score']:.3f})")

if __name__ == "__main__":
    main()
