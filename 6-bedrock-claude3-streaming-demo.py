import json
import boto3
import streamlit as st
import dotenv

dotenv.load_dotenv()

st.title("Amazon Bedrock Claude3 Response Streaming Demo")


client = boto3.client("bedrock-runtime")

# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
model_id = "anthropic.claude-3-haiku-20240307-v1:0"


def parse_stream(stream):
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            message = json.loads(chunk.get("bytes").decode())
            if message['type'] == "content_block_delta":
                yield message['delta']['text'] or ""
            elif message['type'] == "message_stop":
                return "\n"


if prompt := st.text_input("Prompt"):

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        # "system": "You are a helpful assistant",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    })

    streaming_response = client.invoke_model_with_response_stream(
        modelId=model_id,
        body=body,
    )

    st.subheader("Output stream", divider="rainbow")
    stream = streaming_response.get("body")
    st.write_stream(parse_stream(stream))
    st.write_stream(stream)
