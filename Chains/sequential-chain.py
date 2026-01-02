from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

parser = StrOutputParser()

template1=PromptTemplate(
    template='generate a detailed report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='generate a 5 line summary of this report. \n {text}',
    input_variables=['text']
)

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'cricket'})

print(result)

chain.get_graph().print_ascii()