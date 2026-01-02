from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-2.5-flash')

schema=[
    ResponseSchema(name='fact_1', description='fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='fact 3 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template='give 3 facts about the topic {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

prompt=template.format(topic='ninja hattori')

result=model.invoke(prompt)

final_result=parser.parse(result.content)

print(final_result)