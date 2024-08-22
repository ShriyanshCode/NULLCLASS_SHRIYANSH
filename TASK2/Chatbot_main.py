import os
import requests
from pymongo import MongoClient
from googletrans import Translator
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
import streamlit as st
import google.generativeai as genai
from params import MONGO_URI, GOOGLE_API_KEY, HF_TOKEN

client = MongoClient(MONGO_URI)
dbName = "arxiv_papers"  # Adjust as needed
collectionName = "qfin"  # Adjust as needed
collection = client[dbName][collectionName]

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

translator = Translator()

def split_text(text, max_length):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def query_metadata(query):
    user_lang_info = translator.detect(query)
    user_lang = user_lang_info.lang
    user_ip = translator.translate(query, dest='en')
    user_txt = user_ip.text

    docs = vectorStore.similarity_search(user_txt, k=1)
    if not docs:
        return None, None, None

    try:
        doc = docs[0]
        raw_content = doc.page_content
        doc_metadata = doc.metadata
        doc_id = doc_metadata.get('paper_id', None)
        pdf_paper = f"https://arxiv.org/pdf/{doc_id}.pdf" if doc_id else None

        # PDF URL correction if needed
        if pdf_paper:
            arxiv_id_str = str(doc_id).strip()
            if len(arxiv_id_str.split('.')[0]) < 4:  
                arxiv_id_str = '0' + arxiv_id_str
                pdf_paper = f"https://arxiv.org/pdf/{arxiv_id_str}.pdf"

        as_output = raw_content

    except Exception as e:
        print(f"Error: {e}")
        as_output = None
        pdf_paper = None

    llm = GoogleGenerativeAI(model="gemini-1.5-flash-001", google_api_key=GOOGLE_API_KEY)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(user_txt)

    if len(retriever_output) > 1500:
        chunks = split_text(retriever_output, 1500)
        translated_chunks = [translator.translate(chunk, dest=user_lang).text for chunk in chunks]
        final_output = ''.join(translated_chunks)
    else:
        final_output = translator.translate(retriever_output, dest=user_lang).text

    return as_output, final_output, pdf_paper

def summarize_pdf(pdf_url, prompt):
    user_lang_info = translator.detect(prompt)
    user_lang = user_lang_info.lang
    translated_prompt = translator.translate(prompt, dest='en').text

    pdf_path = "temp_pdf.pdf"
    
    # Attempt to download the PDF from the provided URL
    response = requests.get(pdf_url)
    
    if response.status_code != 200:
        arxiv_id = pdf_url.split('/')[-1].replace('.pdf', '')
        arxiv_id_str = str(arxiv_id).strip()
        if len(arxiv_id_str.split('.')[0]) < 4:  
            arxiv_id_str = '0' + arxiv_id_str
            corrected_url = f"https://arxiv.org/pdf/{arxiv_id_str}.pdf"
        
        # Try to download the corrected URL
        response = requests.get(corrected_url)
        if response.status_code == 200:
            pdf_url = corrected_url
        else:
            raise ValueError("The PDF URL does not exist, even after correction.")

    with open(pdf_path, 'wb') as f:
        f.write(response.content)

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    sample_file = genai.upload_file(path=pdf_path, display_name="Uploaded PDF")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    # Generate a answer to query of the document with the given prompt
    response = model.generate_content([sample_file, translated_prompt])
    summary = response.text

    if len(summary) > 1500:
        chunks = split_text(summary, 1500)
        translated_chunks = [translator.translate(chunk, dest=user_lang).text for chunk in chunks]
        final_summary = ''.join(translated_chunks)
    else:
        final_summary = translator.translate(summary, dest=user_lang).text

    return final_summary, pdf_url

# Streamlit UI
st.title("ArXiv Quantitive Finance QueryBot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your question or request:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Processing your request..."):
        if any(keyword in prompt.lower() for keyword in ["lisence", "abstract", "author"]):
            # Extract the metadata using query_metadata
            as_output, final_output, pdf_paper = query_metadata(prompt)
            if final_output:
                st.chat_message("assistant").markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})
                if pdf_paper:
                    st.chat_message("assistant").markdown(f"[View PDF]({pdf_paper})")
                    st.session_state.messages.append({"role": "assistant", "content": f"[View PDF]({pdf_paper})"})
            else:
                st.error("No results found for your query.")
        else:
            pdf_url = prompt if prompt.endswith(".pdf") else None
            if not pdf_url:
                _, _, pdf_paper = query_metadata(prompt)
                pdf_url = pdf_paper

            if pdf_url:
                summary, pdf_link = summarize_pdf(pdf_url, prompt)
                st.chat_message("assistant").markdown(summary)
                st.session_state.messages.append({"role": "assistant", "content": summary})
                st.chat_message("assistant").markdown(f"[View PDF]({pdf_link})")
                st.session_state.messages.append({"role": "assistant", "content": f"[View PDF]({pdf_link})"})
            else:
                st.error("No PDF found for summarization.")

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()
