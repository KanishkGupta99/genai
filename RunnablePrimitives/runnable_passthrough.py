from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser=StrOutputParser()

template1=PromptTemplate(
    template='write a joke about this topic- {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='give explaination of this joke-{joke}',
    input_variables=['joke']
)

joke_gen_chain=RunnableSequence(template1, model, parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(template2, model, parser)
})

chain=RunnableSequence(joke_gen_chain, parallel_chain)

result=chain.invoke({'topic':'Attack on titan'})

print(result['joke'])
print(result['explaination'])

chain.get_graph().print_ascii()
