import streamlit as st
from langchain_community.llms import Bedrock


model_id = 'anthropic.claude-v2'


# Reference: https://docs.streamlit.io/knowledge-base/tutorials/llm-quickstart
st.title('🦜🔗 Text Generation App')


def generate_responses(input_text):
    llm = Bedrock(
        credentials_profile_name="us-east-1",
        model_id=model_id
    )
    st.info(llm(input_text))


with st.form('my_form'):
    text = st.text_area(
        'Enter text:', 'What are the three key pieces of advice for learning how to code?')

    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_responses(text)
