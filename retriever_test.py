from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# Vector Store
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

results = retriever.invoke(
    "What is attention?"
)

print("\nTop Retrieved Chunk:\n")
print(results[0].page_content)