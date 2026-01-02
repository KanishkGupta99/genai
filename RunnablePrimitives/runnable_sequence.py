from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

template1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='Explain this joke- \n {joke}',
    input_variables=['joke']
)

parser=StrOutputParser()

chain=RunnableSequence(template1, model, parser, template2, model, parser)

result=chain.invoke({'topic':'AI'})

print(result)