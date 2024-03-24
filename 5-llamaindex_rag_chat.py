# Reference: https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores import ChromaVectorStore
import chromadb
import streamlit as st
from langchain_community.llms import Bedrock
from langchain_community.embeddings import BedrockEmbeddings

from llama_index.embeddings import LangchainEmbedding
from llama_index import VectorStoreIndex, ServiceContext, set_global_service_context
from llama_index.readers import SimpleWebPageReader


bedrock_embedding = BedrockEmbeddings(
    credentials_profile_name="us-east-1",
    model_id="amazon.titan-embed-text-v1"
)
embed_model = LangchainEmbedding(bedrock_embedding)

model_kwargs = {
    "max_tokens_to_sample": 4096,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman:"],
}
llm = Bedrock(
    credentials_profile_name="us-east-1",
    model_id="anthropic.claude-v2",
    model_kwargs=model_kwargs
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    system_prompt="You are an expert on the LlamaIndex Python library and your job is to answer technical questions. Assume that all questions are related to the LlamaIndex Python library. Keep your answers technical and based on facts – do not hallucinate features."
)
set_global_service_context(service_context)

st.title("Chat with the LlamaIndex Docs💬 📚")

if "message" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant",
            "content": "Ask me a question about LlamaIndex's open-source Python library!"}
    ]


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the LlamaIndex docs – hang tight! This should take 1-2 minutes."):
        # reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        reader = SimpleWebPageReader(html_to_text=True)
        docs = reader.load_data([
            "https://docs.llamaindex.ai/en/stable/examples/data_connectors/WebPageDemo.html"
        ])
        db = chromadb.PersistentClient(path="./llamaindex_chroma_db")
        chorma_collection = db.get_or_create_collection('llamaindex_doc')
        vector_store = ChromaVectorStore(chroma_collection=chorma_collection)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs, storage_context=storage_context)
        return index


index = load_data()
chat_engine = index.as_chat_engine(
    # chat_mode="condense_question", verbose=False)
    chat_mode="condense_question")

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        st.spinner("Thinking...")
        response = chat_engine.chat(prompt).response
        st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)
