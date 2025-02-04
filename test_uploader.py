import streamlit as st
from io import StringIO
import pdfplumber

uploaded_files = st.file_uploader(
    "Choose a CSV file", type= "pdf", accept_multiple_files=True
)



def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

for uploaded_file in uploaded_files:
    st.write(extract_text_from_pdf(uploaded_file))
