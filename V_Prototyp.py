import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA 
from langchain.document_loaders import TextLoader 
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

import streamlit as st
import pdfplumber

#streamlit run c:/Users/letie/spiced/u_rag/V_Prototyp.py [ARGUMENTS]

st.image("C:\\Users\\letie\\spiced\\u_rag\\Ross & Co.png", width=100)




st.title("""Mike Ross AI-RA-Assistent (Model-Presentation) """)




OPENAI_API_KEY= (os.getenv("OPENAI_API_KEY"))

import streamlit as st
from langchain.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader  # Importiere den PDF-Reader

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile

# Datei-Upload
uploaded_files = st.file_uploader(
    "Wähle eine oder mehrere PDF-Dateien aus", type="pdf", accept_multiple_files=True
)

persist_directory = "db4"


# Funktion zur Extraktion von Text aus den hochgeladenen PDF-Dateien
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

embedding = OpenAIEmbeddings()

# Verarbeitung der hochgeladenen Dateien
if uploaded_files:
    documents = []  # Liste zur Speicherung der Dokumente (als Document-Objekte)
    
    for uploaded_file in uploaded_files:
        # Erstelle eine temporäre Datei für jede hochgeladene PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())  # Schreibe die hochgeladene Datei in die temporäre Datei
            temp_file_path = temp_file.name
        
        # Lade das Dokument mit PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()  # Lade das Dokument
        documents.extend(docs)  # Füge die geladenen Dokumente zur Liste hinzu
        #Splitte die Dokumente in kleinere Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
        
    print("text:",texts)  
    #vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

    #retriever = vectordb.as_retriever()

    # define how wide the search should be
    #retriever = vectordb.as_retriever(search_kwargs={ "k": 2})
        
    

    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="result"  # only the result should be stored and shown
    )

    # Den Chatverlauf aus dem Memory abrufen
    chat_history = memory.chat_memory.messages

    # Den Verlauf formatieren und anzeigen
    for message in chat_history:
        if message.type == "human":
            st.write
            (f"User: {message.content}")
        elif message.type == "ai":
            st.write(f"AI: {message.content}")
    


    # qa_chain = RetrievalQA.from_chain_type (llm = OpenAI(),
    #                                 chain_type="stuff", # concetante the question and the answer 
    #                                 retriever=retriever,
    #                                 return_source_documents=True,
    #                                 memory=memory)

    #supplying a persist_directory will store embedding on disk
    
    ## OpenAIEmbeddings
    #embedding = OpenAIEmbeddings()




    # create the retriever

    #retriever = vectordb.as_retriever()

    # define how wide the search should be
    #retriever = vectordb.as_retriever(search_kwargs={ "k": 2})




# make the chain
# create the memory


# the actual chain


    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="result"  # Hier wird festgelegt, dass nur "result" gespeichert wird
    )

#qa_chain = RetrievalQA.from_chain_type (llm = OpenAI(),
#                                       chain_type="stuff", # concetante the question and the answer 
#                                      retriever=retriever,
#                                     return_source_documents=True,
#                                    memory=memory)




# function to show the source of the retrieved documents

    def process_llm_response(llm_response):
        # Ergebnis (Antwort) aus dem LLM extrahieren
        result = f"Antwort: {llm_response['result']}\n\n"
        
        # Quellen sammeln
        sources = "Quellen:\n"
        sources += "\n".join([f"- {source.metadata['source']}" for source in llm_response["source_documents"]])
        
        # Ergebnis und Quellen kombiniert zurückgeben
        return result + sources


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # React to user input
    if prompt := st.chat_input("Stell mir bitte eine Frage zu deinen Fällen!"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)

        retriever = vectordb.as_retriever()

        # define how wide the search should be
        retriever = vectordb.as_retriever(search_kwargs={ "k": 2})
        qa_chain = RetrievalQA.from_chain_type (llm = OpenAI(),
                                    chain_type="stuff", # concetante the question and the answer 
                                    retriever=retriever,
                                    return_source_documents=True,
                                    memory=memory)
    #query = st.text_input('Welche Frage zu Fällen soll ich beantworten?')
        llm_response = qa_chain(prompt)
        #if prompt:
            #for message in chat_history:
                #if message.type == "human":
                #    st.write(f"User: {message.content}")
                #elif message.type == "ai":
                #   st.write(f"AI: {message.content}") 
        answer=process_llm_response(llm_response)
        #st.write(answer)
            

        response = f"KI-Mike Ross {answer}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})




# chatbot ui : https://docs.streamlit.io/develop/tutorials/chat-and-llm-apps/build-conversational-apps

