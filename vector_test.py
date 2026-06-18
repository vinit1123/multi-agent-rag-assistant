from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

print("Chunks:", len(chunks))

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# FAISS Vector Store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

print("FAISS Vector DB Created Successfully!")