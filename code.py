import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings
import os 
from dotenv import load_dotenv

load_dotenv()

# Input text from user via Streamlit
text = st.text_input("Enter text to embed:", "HELLO , MARWADI UNIVERSITY")

# Button to trigger embedding
if st.button("Generate Embedding"):
    # Create embedding model
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("Embedding_AZURE_OPENAI_ENDPOINT"),
        openai_api_key=os.getenv("Embedding_AZURE_OPENAI_API_KEY"),
        deployment=os.getenv("Embedding_AZURE_OPENAI_DEPLOYMENT_NAME"),
        chunk_size=10,
    )

    # Generate embedding
    embedding = embedding_model.embed_query(text)

    # Output using Streamlit
    st.success("✅ Embedding generated successfully!")
    st.write(embedding)
    st.write("📏 Length of embedding vector:", len(embedding))