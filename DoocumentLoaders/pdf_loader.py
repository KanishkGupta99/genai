from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('/Users/kanishkgupta/Desktop/genai/DoocumentLoaders/Atlassian_Intern_Interview_Guide_-_Engineering (1).pdf')

docs=loader.load()

print(docs)
print(len(docs))
