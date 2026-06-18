from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/sample.pdf")

docs = loader.load()

print("Pages:", len(docs))
print("\nFirst Page Content:\n")
print(docs[0].page_content[:1000])