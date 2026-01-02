from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('/Users/kanishkgupta/Desktop/genai/DoocumentLoaders/Atlassian_Intern_Interview_Guide_-_Engineering (1).pdf')

docs=loader.load() 
splitter=CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

print(len(docs))

result=splitter.split_documents(docs)

print(result[0])
print(result[1])