import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Paths
DOCUMENT_PATH = Path("knowledge_base/customer_support.txt")
VECTOR_STORE_PATH = Path("vector_store")

# Check document
if not DOCUMENT_PATH.exists():
    print("ERROR: Knowledge-base file not found.")
    exit()

print("Loading knowledge document...")

# Load document
loader = TextLoader(
    str(DOCUMENT_PATH),
    encoding="utf-8"
)

documents = loader.load()

print(f"Document loaded successfully: {len(documents)} document(s)")

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# Create embeddings
print("Creating OpenAI embeddings...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Create FAISS vector store
print("Creating FAISS vector store...")

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)

# Save locally
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

vector_store.save_local(str(VECTOR_STORE_PATH))

print("\nFAISS vector store created successfully!")
print(f"Location: {VECTOR_STORE_PATH.absolute()}")