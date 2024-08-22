import json
import os
from pymongo import MongoClient
from googletrans import Translator, LANGUAGES
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
import streamlit as st
from params import MONGO_URI, GOOGLE_API_KEY, HF_TOKEN

client = MongoClient(MONGO_URI)
dbName = "medquad_1k"
collectionName = "qna_collections"
collection = client[dbName][collectionName]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

translator = Translator()

def split_text(text, max_length):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def query_data(query):
    try:
        # Extracting language info from query
        user_lang_info = translator.detect(query)
        user_lang = user_lang_info.lang
        
        # Handling the case where language detection fails
        if user_lang not in LANGUAGES:
            raise ValueError("Language not detected.")
        
        user_ip = translator.translate(query, dest='en')
        user_txt = user_ip.text
    except Exception as e:
        st.error(f"Error in language detection or translation: {str(e)}")
        return None, None

    docs = vectorStore.similarity_search(user_txt, k=1)
    if not docs:
        return None, None

    try:
        doc_content = json.loads(docs[0].page_content)
        as_output = doc_content['text']
    except (json.JSONDecodeError, KeyError):
        as_output = docs[0].page_content

    llm = GoogleGenerativeAI(model="gemini-1.5-flash-001", google_api_key=GOOGLE_API_KEY)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(user_txt)
    
    # Translate API has a cap of 1.5k characters sent at once, split into multiple chunks if answer is more than 1.5k chunks
    if len(retriever_output) > 1500:
        chunks = split_text(retriever_output, 1500)
        translated_chunks = [translator.translate(chunk, dest=user_lang).text for chunk in chunks]
        final_output = ''.join(translated_chunks)
    else:
        final_output = translator.translate(retriever_output, dest=user_lang).text

    return as_output, final_output

# Streamlit UI
st.title("MedQuAD Q&A Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a medical question:"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Adding spinner while processing the query
    with st.spinner("Processing query..."):
        as_output, final_output = query_data(prompt)

    if final_output:
        with st.chat_message("assistant"):
            st.markdown(final_output)
        st.session_state.messages.append({"role": "assistant", "content": final_output})
    else:
        st.error("No results found for your query.")

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
