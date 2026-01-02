from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Prompt 1 -> report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# Prompt 2 -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text with structured output\n{text}",
    input_variables=["text"]
)

# # -------- Step 1: Generate report --------
# prompt1 = template1.format(topic="black hole")
# result1 = model.invoke(prompt1)

# print("REPORT:\n", result1.content)

# # -------- Step 2: Generate summary --------
# prompt2 = template2.format(text=result1.content)
# result2 = model.invoke(prompt2)

# print("\nSUMMARY:\n", result2.content)


parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)