from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import PromptTemplate,load_prompt

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header('🎌 Anime Summariser')

anime_name = st.selectbox(
    'Select the name of the anime',
    ['Death Note', 'Attack On Titan', 'Demon Slayer', 'One Punch Man', 'Haikyuu']
)

style_input = st.selectbox(
    'Select the style of explanation',
    ['Story Wise', 'Genre Wise', 'Character Wise', 'Popularity Wise']
)

explanation_length = st.selectbox(
    'Select the length of explanation',
    ['Short', 'Medium', 'Long']
)

template=load_prompt('template.json')

if st.button('Summarise'):
    user_input=template.format(
        anime_name=anime_name,
        style_input=style_input,
        explanation_length=explanation_length
    )
    with st.spinner('Generating summary...'):
        output = model.invoke(user_input)

    st.subheader('Summary:')
    st.write(output.content)
