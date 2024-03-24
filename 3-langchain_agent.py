from langchain_community.llms import Bedrock
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.chat_models.bedrock import BedrockChat
from langchain.agents import AgentType, initialize_agent, load_tools


# Reference: https://python.langchain.com/docs/integrations/callbacks/streamlit
st.title('🦜🔗 Agent & Streaming App')


model_id = 'anthropic.claude-v2'

st_callback = StreamlitCallbackHandler(st.container())

llm = BedrockChat(
    credentials_profile_name='us-east-1',
    model_id=model_id,
    streaming=True
)


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


tools = load_tools(['ddg-search', 'llm-math'], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True,
    handle_parsing_errors=True
)


if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        response = agent.run(prompt, callbacks=[
                             stream_handler])
        st.write(response)
