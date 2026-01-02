from langchain_community.document_loaders import WebBaseLoader

url='https://medium.com/@shravankoninti/multimodal-rag-with-gpt-4-vision-and-langchain-60a6a13a92e4'
loader=WebBaseLoader(url)

docs=loader.load()

print(docs[0].page_content)