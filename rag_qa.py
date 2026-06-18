from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
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

# LLM
llm = ChatOllama(
    model="llama3.2"
)

question = "What is Multi Head Attention?"

# Retrieve context
docs = retriever.invoke(question)

context = "\n\n".join(
    doc.page_content for doc in docs
)

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response.content)