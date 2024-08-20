import streamlit as st  # Importing the streamlit library
from langchain_openai import OpenAI  # Importing the OpenAI class from langchain_openai module
from langchain.chains.summarize import load_summarize_chain  # Importing the load_summarize_chain function from langchain.chains.summarize module
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd  # Importing the pandas library
from io import StringIO  # Importing the StringIO class from the io module

## function to load LLM 
def load_llm(openai_api_key):
    llm = OpenAI(openai_api_key=openai_api_key)
    return llm

## page title and header 
st.set_page_config(
    page_title="AI Long Text Summarizer",
    page_icon="🔗",
    layout="centered",
)
st.header("AI Long Text Summarizer")

## intro text
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
       ChatGPT cannot summarize long texts. Now you can do it with this app.
        """
    )

with col2:
    st.write("Contact: [Email](mailto:tabdulazeez99@gmail.com) to build your AI project" )

## input api key 
openai_api_key = st.sidebar.text_input("Enter your OpenAI API key",placeholder="Enter your OpenAI API key",key="openai_api_key", type="password")


## input 
st.markdown('## Upload the file to summarize')
uploaded_file = st.file_uploader("Choose a file", type='txt')

## output
st.markdown('### Summarized Text:')

if uploaded_file is not None:
    # Read the file
    bytes_data = uploaded_file.getvalue()

    # Convert the bytes data to string
    stringio = StringIO(bytes_data.decode('utf-8'))

    # Read the string data
    string_data = stringio.read()

    file_input = string_data

    if len(file_input.split(" ")) > 20000:
        st.write("Text is too long. Please upload a text with less than 20000 words.")
        st.stop()
    if file_input:
        if not openai_api_key:
            st.write("Please enter your OpenAI API key in the sidebar.")
            st.stop()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n',"\n"],
        chunk_size=5000,
        chunk_overlap=350,
    )

    splitted_documents = text_splitter.create_documents(file_input)

    llm = load_llm(openai_api_key=openai_api_key)

    summarize_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce"
       
    )

    summarized_text = summarize_chain.run(splitted_documents)
    st.write(summarized_text)

