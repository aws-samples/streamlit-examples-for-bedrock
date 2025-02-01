import json
import boto3
import streamlit as st

# App configuration
st.set_page_config(page_title="Amazon Bedrock Chat", layout="wide")
st.title("💬 Amazon Bedrock Chat")
st.caption("🚀 Powered by Nova via Amazon Bedrock")

# Initialize Bedrock client using Streamlit secrets
try:
    client = boto3.client(
        "bedrock-runtime",
        region_name=st.secrets.AWS["AWS_DEFAULT_REGION"],
        aws_access_key_id=st.secrets.AWS["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=st.secrets.AWS["AWS_SECRET_ACCESS_KEY"]
    )
except Exception as e:
    st.error(f"Failed to initialize Bedrock client: {str(e)}")
    st.stop()

# Model configuration
# MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
MODEL_ID = "us.amazon.nova-lite-v1:0"
REGION = "us-east-1"

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for system prompt
with st.sidebar:
    st.header("Configuration")
    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful AI assistant.",
        help="Define the AI's behavior and personality"
    )
    st.divider()
    st.markdown("**Model Info**")
    st.text(f"Model: {MODEL_ID}")
    st.text(f"Region: {REGION}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare the request body
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([{"role": m["role"], "content": m["content"]} for m in st.session_state.messages])

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages
    })

    # Stream the response
    try:
        response = client.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=body
        )
        
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            for event in response.get("body"):
                chunk = event.get('chunk')
                if chunk:
                    message = json.loads(chunk.get("bytes").decode())
                    if message['type'] == "content_block_delta":
                        text = message['delta']['text'] or ""
                        full_response += text
                        response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")