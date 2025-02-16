import os
import argparse
import openai
from pinecone import Pinecone, ServerlessSpec
import tiktoken  # Ensure you have tiktoken installed (pip install tiktoken)
import sys

# Optional libraries for file processing
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None

try:
    import docx  # for processing .docx files
except ImportError:
    docx = None

# -------------------- CONFIGURATION --------------------
def load_config():
    pinecone_key = os.getenv("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))
    openai_key = os.getenv("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    
    if not pinecone_key or not openai_key:
        raise EnvironmentError("Please set both PINECONE_API_KEY and OPENAI_API_KEY environment variables.")
    return pinecone_key, openai_key

# -------------------- ARGUMENT PARSING --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run Pinecone Embedding Pipeline with interactive or non-interactive mode")
    parser.add_argument("--index", type=str, help="Name of the Pinecone index to use or create")
    parser.add_argument("--namespace", type=str, help="Namespace to use (if omitted, will prompt or use default)")
    parser.add_argument("--directory", type=str, help="Path to the directory containing files")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode using provided defaults")
    return parser.parse_args()

# -------------------- INTERACTIVE INDEX SELECTION --------------------
def select_or_create_index(pc, provided_index=None, non_interactive=False):
    existing_indexes = list(pc.list_indexes().names())
    if provided_index:
        if provided_index in existing_indexes:
            print(f"Using provided existing index: {provided_index}")
            return provided_index
        else:
            print(f"Provided index '{provided_index}' does not exist. Creating it...")
            pc.create_index(
                name=provided_index,
                dimension=3072,  # Adjust based on OpenAI model used
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            return provided_index

    if non_interactive:
        # Fallback default in non-interactive mode
        default_index = "default_index"
        if default_index not in existing_indexes:
            print(f"Creating default index '{default_index}'...")
            pc.create_index(
                name=default_index,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        else:
            print(f"Using existing default index: {default_index}")
        return default_index

    # Interactive mode
    print("Existing indexes:", existing_indexes)
    while True:
        choice = input("Do you want to (s)elect an existing index or (c)reate a new one? [s/c]: ").strip().lower()
        if choice not in ("s", "c"):
            print("Invalid choice. Please enter 's' to select or 'c' to create.")
            continue
        if choice == "s":
            index_name = input("Enter the name of the existing index: ").strip()
            if index_name in existing_indexes:
                return index_name
            else:
                create_choice = input(f"Index '{index_name}' does not exist. Would you like to create it instead? (y/n): ").strip().lower()
                if create_choice == "y":
                    print(f"Creating new index '{index_name}'...")
                    pc.create_index(
                        name=index_name,
                        dimension=3072,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                    )
                    return index_name
                else:
                    print("Let's try again.")
                    continue
        elif choice == "c":
            index_name = input("Enter a name for the new index: ").strip()
            if index_name in existing_indexes:
                print(f"Index '{index_name}' already exists. Using the existing index.")
            else:
                print(f"Creating new index '{index_name}'...")
                pc.create_index(
                    name=index_name,
                    dimension=3072,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            return index_name

# -------------------- INTERACTIVE NAMESPACE SELECTION --------------------
def select_or_create_namespace(provided_namespace=None, non_interactive=False):
    if provided_namespace is not None:
        print(f"Using provided namespace: '{provided_namespace}'")
        return provided_namespace
    if non_interactive:
        default_namespace = "default_namespace"
        print(f"Non-interactive mode: using default namespace '{default_namespace}'")
        return default_namespace

    namespace = input("Enter a namespace to use (or press Enter to use the default namespace): ").strip()
    return namespace if namespace != "" else None

# -------------------- FILE PROCESSING & TOKENIZATION --------------------
def tokenize_text(text):
    # Use the ideal tokenizer for OpenAI models; "cl100k_base" is common for many models.
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return tokens

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
                    text += page.extract_text() or ""
        elif ext in ['.docx'] and docx:
            doc = docx.Document(filepath)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            # For unknown types, attempt to read as text
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
    tokens = tokenize_text(text)
    print(f"Processed '{filepath}' with {len(tokens)} tokens.")
    return text

def process_directory(directory):
    documents = []
    doc_id = 1
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            text = process_file(filepath)
            if text.strip():
                documents.append({
                    "id": f"doc{doc_id}",
                    "text": text
                })
                doc_id += 1
    return documents

# -------------------- MAIN FUNCTION --------------------
def main():
    args = parse_args()
    # Load API keys
    PINECONE_API_KEY, OPENAI_API_KEY = load_config()

    # Initialize OpenAI API client
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    # Initialize Pinecone Client (New API)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Determine index name using provided argument or interactive/default mode
    index_name = select_or_create_index(pc, provided_index=args.index, non_interactive=args.non_interactive)
    index = pc.Index(index_name)

    # Determine namespace using provided argument or interactive/default mode
    namespace = select_or_create_namespace(provided_namespace=args.namespace, non_interactive=args.non_interactive)

    # -------------------- FUNCTION: EMBED TEXT --------------------
    def embed_text(texts):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return [r.embedding for r in response.data]

    # -------------------- FUNCTION: UPSERT DATA --------------------
    def upsert_data(documents):
        embeddings = embed_text([doc["text"] for doc in documents])
        vectors = [
            {
                "id": doc["id"],
                "values": emb,
                "metadata": {"text": doc["text"]}
            }
            for doc, emb in zip(documents, embeddings)
        ]
        ns = namespace if namespace else ""
        index.upsert(vectors, namespace=ns)
        print(f"Upserted {len(vectors)} documents into index '{index_name}'" +
              (f" under namespace '{ns}'." if ns else "."))

    # -------------------- FUNCTION: QUERY PINECONE --------------------
    def query_pinecone(query, top_k=3):
        query_embedding = embed_text([query])[0]
        ns = namespace if namespace else ""
        results = index.query(
            namespace=ns,
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return results

    # -------------------- PIPELINE EXECUTION --------------------
    # Determine directory
    if args.directory:
        directory = args.directory
    elif args.non_interactive:
        # Provide a default directory for non-interactive mode
        directory = "./documents"
        print(f"Non-interactive mode: using default directory '{directory}'")
    else:
        directory = input("Enter the path to the directory containing your files: ").strip()

    if not os.path.isdir(directory):
        print(f"Directory '{directory}' is invalid. Exiting.")
        sys.exit(1)

    print("Processing files in directory. This may take a while...")
    documents = process_directory(directory)
    if not documents:
        print("No valid documents found. Exiting.")
        sys.exit(1)

    upsert_data(documents)

    # Interactive query loop (only if not in non-interactive mode)
    if not args.non_interactive:
        while True:
            choice = input("Would you like to (q)uery the index or (e)xit? [q/e]: ").strip().lower()
            if choice == "q":
                user_query = input("Enter your query: ").strip()
                results = query_pinecone(user_query)
                print("\nTop Results:")
                for match in results["matches"]:
                    print(f"- {match['metadata']['text']} (Score: {match['score']:.4f})")
            elif choice == "e":
                print("Exiting. Goodbye!")
                break
            else:
                print("Invalid option. Please enter 'q' to query or 'e' to exit.")
    else:
        print("Non-interactive mode complete. Exiting.")

if __name__ == "__main__":
    main()