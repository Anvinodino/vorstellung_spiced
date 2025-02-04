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
import shutil

#streamlit run c:/Users/letie/spiced/u_rag/V_Prototyp.py [ARGUMENTS]

st.image("C:\\Users\\letie\\spiced\\u_rag\\Ross & Co.png", width=100)




st.title("""Mike Ross AI-RA-Assistent (Vorführmodell) """)




OPENAI_API_KEY= (os.getenv("OPENAI_API_KEY"))

import streamlit as st
from langchain.document_loaders import PyPDFLoader

from PyPDF2 import PdfReader  # Importiere den PDF-Reader

from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import tempfile

import json
from llama_index.readers.google import GoogleDriveReader

# Lade die Service-Account-Schlüsseldatei
with open("credentials.json") as f:
    service_account_key = json.load(f)

# Initialize the reader
reader = GoogleDriveReader(
    folder_id="1omQKaU6syLr5X0kcr1_Vnl0LKNfQzanI",
    service_account_key=service_account_key,  # Hier das geladene Dict übergeben!
)

if st.button('Lade Dokumente'):
    new_documents = reader.load_data()
    # Wenn du Duplikate vermeiden willst, kannst du einen Hash oder die Datei-ID als Schlüssel verwenden.
    # # Beispiel: Setzen eines Sets, um IDs zu speichern und Duplikate zu vermeiden
    
    unique_documents = []
    seen_ids = set()  # Hier speichern wir bereits gesehene IDs
    for doc in new_documents:
        doc_id = getattr(doc, "doc_id", None)  # Annahme: Jedes Dokument hat eine ID
        if doc_id not in seen_ids:
            unique_documents.append(doc)
            seen_ids.add(doc_id)
            print(f"Anzahl eindeutiger Dokumente: {len(unique_documents)}")

from langchain.schema import Document

# Neue Liste mit korrekten `Document`-Objekten erstellen
corrected_documents = [
    Document(page_content=doc.text_resource.text, metadata=doc.metadata) for doc in new_documents
]




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

#supplying a persist_directory will store embedding on disk
persist_directory = "db1"

if os.path.exists(persist_directory):
    # Verzeichnis löschen
    shutil.rmtree(persist_directory)
    print(f"Das Verzeichnis {persist_directory} wurde gelöscht.")
else:
    print(f"Das Verzeichnis {persist_directory} existiert nicht.")

## OpenAIEmbeddings



text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(corrected_documents)


embedding = OpenAIEmbeddings()

vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=persist_directory)


# create the retriever

retriever = vectordb.as_retriever()

# define how wide the search should be
retriever = vectordb.as_retriever(search_kwargs={ "k": 2})


def generate_multi_query(query, model="gpt-3.5-turbo"):

    prompt = """
Du bist ein sachkundiger Rechtsanwaltsassistent für die Recherche.
Deine Nutzer stellen viele Anfragen zu Briefen und Gerichtsakten.
Für die gegebene Frage schlage bis zu 5 verwandte Fragen vor, um ihnen zu helfen, die benötigten Informationen zu finden.
Stelle prägnante, themenbezogene Fragen (ohne zusammengesetzte Sätze), die verschiedene Aspekte des Themas abdecken.
Stelle sicher, dass jede Frage vollständig und direkt mit der ursprünglichen Anfrage verbunden ist.
Liste jede Frage in einer eigenen Zeile ohne Nummerierung.
                """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


original_query = "Was hat Bendig am Ende seines Falles bekommen?"
aug_queries = generate_multi_query(original_query)

# 3. Erweiterten Query-String für die Dokumentensuche erstellen
expanded_query = " ".join([original_query] + aug_queries)


retriever = vectordb.as_retriever()

retriever = vectordb.as_retriever(search_kwargs={ "k": 4})







# make the chain
# create the memory


# the actual chain


memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True,
    output_key="result"  # Hier wird festgelegt, dass nur "result" gespeichert wird
)


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Definiere dein individuelles Prompt-Template
custom_prompt = PromptTemplate(
    input_variables=["context", "original_query"],
    template="Beantworte die folgende Frage basierend auf dem gegebenen Kontext:\n\n"
             "Kontext:\n{context}\n\n"
             "Frage: {original_query}\n\n"
             "Antwort:"
)

# RetrievalQA mit dem benutzerdefinierten Prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever= retriever,
    return_source_documents=True,
    memory=memory,
    chain_type_kwargs={"prompt": custom_prompt,
                       "output_key": "result"}  # Hier wird das Prompt eingebunden
)








# function to show the source of the retrieved documents

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nQuellen:')
    for source in llm_response["source_documents"]:
        print(source.metadata.get('file path', 'Unbekannte Quelle'))


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

#query = st.text_input('Welche Frage zu Fällen soll ich beantworten?')
llm_response = qa_chain(prompt)
if prompt:
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

