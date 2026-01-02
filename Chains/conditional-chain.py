from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

class FeedBack(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='give the sentiment of the feedback')


parser1 = PydanticOutputParser(pydantic_object=FeedBack)
parser2=StrOutputParser()

text="""
Phone Review: XYZ Pro

The **XYZ Pro** delivers a solid all-round experience at its price point. The **6.6-inch AMOLED display** is bright and vibrant, making videos and scrolling enjoyable. Performance is smooth for daily tasks and casual gaming, thanks to the efficient processor and ample RAM.

The **camera setup** performs well in good lighting, capturing sharp photos with accurate colors, though low-light shots could be better. Battery life is reliable, easily lasting a full day with moderate use, and **fast charging** is a welcome addition.

On the downside, the phone lacks a headphone jack and the camera app can feel slightly slow at times. Overall, the XYZ Pro is a **great value-for-money phone** for users looking for strong performance, a good display, and dependable battery life. a really good phone.
"""

template1=PromptTemplate(
    template='classify the sentiment of the following feedback into positive or negative- \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser1.get_format_instructions()}
)

template2=PromptTemplate(
    template='Write a appropriate response to the following positive feedback- \n {feedback}',
    input_variables=['feedback']
)

template3=PromptTemplate(
    template='Write a appropriate response to the following negative feedback- \n {feedback}',
    input_variables=['feedback']
)

classifier_chain=template1 | model | parser1

branch_chain=RunnableBranch(
    (lambda x: x.sentiment=='positive',template2 | model | parser2),
    (lambda x: x.sentiment=='negative',template3 | model | parser2),
    RunnableLambda(lambda x:'could not find sentiment')
)

chain=classifier_chain | branch_chain

result=chain.invoke({'feedback':text})

print(result)

chain.get_graph().print_ascii()