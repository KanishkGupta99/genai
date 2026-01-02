from langchain_community.document_loaders import TextLoader

loader=TextLoader('/Users/kanishkgupta/Desktop/genai/DoocumentLoaders/ai.txt',encoding='utf-8')

docs=loader.load()

print(docs[0])
print(type(docs))