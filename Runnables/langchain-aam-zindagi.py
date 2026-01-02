import random

class NakliLLLM:
    def __init__(self):
        print('LLM created')

    def predict(self, prompt):
        response_list=['delhi is the capital of india',
                       'ipl is a cricket tournament',
                       'ai stands for artificial intelligence']
        
        return {'response':random.choice(response_list)}

class nakliPromptTemplate:
    def __init__(self,template,input_variables):
        self.template=template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.template.format(**input_dict)

class nakliChain:
    def __init__(self, llm, template):
        self.llm=llm
        self.template=template
    
    def invoke(self,input_dict):
        prompt=self.template.format(input_dict)
        return llm.predict(prompt)

llm=NakliLLLM()

template=nakliPromptTemplate(
    template='write a poem about {topic}',
    input_variables=['topic']
)

chain=nakliChain(llm,template)

# prompt=template.format({'topic':'india'})

# print(llm.predict(prompt))

print(chain.invoke({'topic':'india'}))