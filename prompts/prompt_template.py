from langchain.prompts import PromptTemplate

template=PromptTemplate(
    input_variables=['anime_name','style_input','explanation_length'],
    template='Summarize the anime {anime_name} in a {style_input} manner. Keep the explanation {explanation_length} in length.',
    validate_template=True
)

template.save('template.json')