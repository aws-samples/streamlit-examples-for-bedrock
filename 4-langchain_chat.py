
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.chat_models.bedrock import BedrockChat
# from langchain.chat_models.anthropic import ChatAnthropic
import streamlit as st

from dotenv import load_dotenv


# Reference: https://python.langchain.com/docs/integrations/memory/streamlit_chat_message_history
st.title("📖 StreamlitChatMessageHistory")

model_id = 'anthropic.claude-v2'
# Load AWS Profile/credentials from .env
load_dotenv()
llm = BedrockChat(
    model_id=model_id,
    streaming=True
)

msgs = StreamlitChatMessageHistory(key='langchain_messages')
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    config = {"configurable": {"session_id": "special_key"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
