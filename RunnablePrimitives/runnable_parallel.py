from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser=StrOutputParser()

template1=PromptTemplate(
    template='write a tweet on this topic- {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='write a linkedin post on this topic- {topic}',
    input_variables=['topic']
)

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(template1, model, parser),
    'linkedin':RunnableSequence(template2, model, parser)
})

result=parallel_chain.invoke({'topic':'AI'})

print(result['tweet'])
print(result['linkedin'])