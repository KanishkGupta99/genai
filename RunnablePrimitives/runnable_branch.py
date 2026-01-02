from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnableLambda

load_dotenv()

class query(BaseModel):
    type:Literal['complain','refund','general']=Field(description='the type of user query')

text="""
“Is this phone good for daily use?”

“Should I buy this phone or look for alternatives?”"""

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')
parser1=PydanticOutputParser(pydantic_object=query)
parser2=StrOutputParser()

template1=PromptTemplate(
    template='give me the type of user query from this user query-{text} \n {format_instruction}',
    input_variables=['text'],
    partial_variables={'format_instruction':parser1.get_format_instructions}
)

template2=PromptTemplate(
    template='give an appropriate response from the follwoing user complaint- {text}',
    input_variables=['text']
)

template3=PromptTemplate(
    template='give an appropriate response from the follwoing user refund query- {text}',
    input_variables=['text']
)

template4=PromptTemplate(
    template='give an appropriate response from the follwoing user general_query- {text}',
    input_variables=['text']
)

classifier_chain=RunnableSequence(template1, model, parser1)

branch_chain=RunnableBranch(
    (RunnableLambda(lambda x:x.type=='complain'),RunnableSequence(template2,model,parser2)),
    (RunnableLambda(lambda x:x.type=='refund'),RunnableSequence(template3,model,parser2)),
    (RunnableLambda(lambda x:x.type=='general'),RunnableSequence(template4,model,parser2)),
    RunnableLambda(lambda x:"could not find query type")
)

chain=RunnableSequence(classifier_chain,branch_chain)

result=chain.invoke({'text':text})

print(result)