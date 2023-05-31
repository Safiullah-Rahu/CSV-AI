import streamlit as st
from streamlit_chat import message
import os, tempfile, sys
from io import BytesIO
from io import StringIO
import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms.openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain


st.set_page_config(page_title="CSV AI", layout="wide")


def home_page():
    st.write("""Select any one feature from above sliderbox: \n
    1. Chat with CSV \n
    2. Summarize CSV \n
    3. Analyze CSV  """)

def chat(temperature, model_name):
    st.write("# Talk to CSV")
    # Add functionality for Page 1
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")

    if uploaded_file :
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            data = loader.load()
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            data = loader.load()

        embeddings = OpenAIEmbeddings()
        vectors = FAISS.from_documents(data, embeddings)
        llm = ChatOpenAI(temperature=temperature, model_name=model_name) # 'gpt-3.5-turbo',
        qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff", 
                                     retriever=vectors.as_retriever(), 
                                     verbose=True)
        #chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=temperature, model_name=model_name), retriever=vectors.as_retriever())

        def conversational_chat(query):
        
#             result = chain({"question": query, "chat_history": st.session_state['history']})
#             st.session_state['history'].append((query, result["answer"]))
            result = qa.run(query) #chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result))#["answer"]))
        
            return result#["answer"]
    
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generatedd' not in st.session_state:
            st.session_state['generatedd'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'pastt' not in st.session_state:
            st.session_state['pastt'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                
                st.session_state['pastt'].append(user_input)
                st.session_state['generatedd'].append(output)

        if st.session_state['generatedd']:
            with response_container:
                for i in range(len(st.session_state['generatedd'])):
                    message(st.session_state["pastt"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                    message(st.session_state["generatedd"][i], key=str(i), avatar_style="bottts")
        if reset:
            st.session_state["pastt"] = []
            st.session_state["generatedd"] = []

def summary(model_name, temperature, top_p, freq_penalty):
    st.write("# Summary of CSV")
    st.write("Upload your document here:")
    uploaded_file = st.file_uploader("Upload source document", type="csv", label_visibility="collapsed")
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        # encoding = cp1252
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=0)
        try:
            loader = CSVLoader(file_path=tmp_file_path, encoding="cp1252")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)
        except:
            loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8")
            #loader = UnstructuredFileLoader(tmp_file_path)
            data = loader.load()
            texts = text_splitter.split_documents(data)

        os.remove(tmp_file_path)
        gen_sum = st.button("Generate Summary")
        if gen_sum:
            # Initialize the OpenAI module, load and run the summarize chain
            llm = OpenAI(model_name=model_name, temperature=temperature)
            chain = load_summarize_chain(llm, chain_type="stuff")
            #search = docsearch.similarity_search(" ")
            summary = chain.run(input_documents=texts[:50])

            st.success(summary)


def analyze(temperature, model_name):
    st.write("# Analyze CSV")
    #st.write("This is Page 3")
    # Add functionality for Page 3
    reset = st.sidebar.button("Reset Chat")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV here 👇:", type="csv")
    #.write(uploaded_file.name)
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        df = pd.read_csv(tmp_file_path)

        def agent_chat(query):

            # Create and run the CSV agent with the user's query
            try:
                agent = create_pandas_dataframe_agent(ChatOpenAI(temperature=temperature, model_name=model_name), df, verbose=True, max_iterations=4)
                result = agent.run(query)
            except:
                result = "Try asking quantitative questions about structure of csv data!"
            return result
   

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about Document 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]
            
        #container for the chat history
        response_container = st.container()
        #container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                
                user_input = st.text_input("Use CSV agent for precise information about the structure of your csv file:", placeholder="e-g : how many rows in my file ?:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = agent_chat(user_input)
                
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
        if reset:
            st.session_state["past"] = []
            st.session_state["generated"] = []


# Main App
def main():
    st.markdown(
        """
        <div style='text-align: center;'>
            <h1>🧠 CSV AI</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style='text-align: center;'>
            <h4>⚡️ Interacting, Analyzing and Summarizing CSV Files!</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


    if os.path.exists(".env") and os.environ.get("OPENAI_API_KEY") is not None:
        user_api_key = os.environ["OPENAI_API_KEY"]
        st.success("API key loaded from .env", icon="🚀")
    else:
        user_api_key = st.sidebar.text_input(
            label="#### Enter OpenAI API key 👇", placeholder="Paste your openAI API key, sk-", type="password", key="openai_api_key"
        )
        if user_api_key:
            st.sidebar.success("API key loaded", icon="🚀")

    os.environ["OPENAI_API_KEY"] = user_api_key

    

    # Execute the home page function
    MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4", "gpt-4-32k"]
    max_tokens = {"gpt-4":7000, "gpt-4-32k":31000, "gpt-3.5-turbo":3000}
    TEMPERATURE_MIN_VALUE = 0.0
    TEMPERATURE_MAX_VALUE = 1.0
    TEMPERATURE_DEFAULT_VALUE = 0.9
    TEMPERATURE_STEP = 0.01
    model_name = st.sidebar.selectbox(label="Model", options=MODEL_OPTIONS)
    top_p = st.sidebar.slider("Top_P", 0.0, 1.0, 1.0, 0.1)
    freq_penalty = st.sidebar.slider("Frequency Penalty", 0.0, 2.0, 0.0, 0.1)
    temperature = st.sidebar.slider(
                label="Temperature",
                min_value=TEMPERATURE_MIN_VALUE,
                max_value=TEMPERATURE_MAX_VALUE,
                value=TEMPERATURE_DEFAULT_VALUE,
                step=TEMPERATURE_STEP,)

    # Define a dictionary with the function names and their respective functions
    functions = [
        "home",
        "Chat with CSV",
        "Summarize CSV",
        "Analyze CSV",
    ]

    #st.subheader("Select any generator👇")
    # Create a selectbox with the function names as options
    selected_function = st.selectbox("Select a functionality", functions)
    if selected_function == "home":
        home_page()
    elif selected_function == "Chat with CSV":
        chat(temperature=temperature, model_name=model_name)
    elif selected_function == "Summarize CSV":
        summary(model_name=model_name, temperature=temperature, top_p=top_p, freq_penalty=freq_penalty)
    elif selected_function == "Analyze CSV":
        analyze(temperature=temperature, model_name=model_name)
    else:
        st.warning("You haven't selected any AI Functionality!!")
    

    st.write("---")
    st.write("Made with ❤️")

if __name__ == "__main__":
    main()
