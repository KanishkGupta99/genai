import random
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input_data):
        pass
 

class NakliLLLM(Runnable):
    def __init__(self):
        print('LLM created')

    def invoke(self, prompt):
        response_list = [
            'delhi is the capital of india',
            'ipl is a cricket tournament',
            'ai stands for artificial intelligence'
        ]
        return random.choice(response_list)


class NakliPromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)
    

class RunnableConnector(Runnable):
    def __init__(self, runnable_list):
        self.runnable_list = runnable_list
    
    def invoke(self, input_data):
        for runnable in self.runnable_list:
            input_data = runnable.invoke(input_data)
        return input_data


llm = NakliLLLM()

template = NakliPromptTemplate(
    template='write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)

# ✅ Correct order: template → llm
chain = RunnableConnector([template, llm])

print(chain.invoke({'length': 'small', 'topic': 'india'}))