import json
import os
import re
from pymongo import MongoClient
from googletrans import Translator
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.chains import RetrievalQA
from params import MONGO_URI, GOOGLE_API_KEY, HF_TOKEN
client = MongoClient(MONGO_URI)
dbName = "arxiv_papers"
collectionName = "qfin"
collection = client[dbName][collectionName]

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorStore = MongoDBAtlasVectorSearch(collection, embeddings)

translator = Translator()

def split_text(text, max_length):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]
document = collection.find_one()  
print(document)

def query_data(query):
    user_lang_info = translator.detect(query)
    user_lang = user_lang_info.lang
    print(f"Detected language: {user_lang}")

    user_ip = translator.translate(query, dest='en')
    user_txt = user_ip.text
    print(f"Translated query to English: {user_txt}")

    # Perform vector search with translated query
    docs = vectorStore.similarity_search(user_txt, k=1)
    if not docs:
        print("No documents found matching the query.")
        return None, None, None  
    try:
        doc = docs[0]
        raw_content = doc.page_content
        doc_metadata = doc.metadata  
        # Print raw document content
        print(f"Raw document content: {raw_content}")

        # Extract id from document metadata
        doc_id = doc_metadata.get('paper_id', None)  # Adjust key name based on metadata structure
        if doc_id is not None:
            pdf_paper = f"https://arxiv.org/pdf/{doc_id}"
            print(f"Extracted PDF paper URL: {pdf_paper}")
        else:
            pdf_paper = None
            print("ID not found in document metadata.")
        as_output = raw_content

    except Exception as e:
        print(f"Unexpected error: {e}")
        as_output = None
        pdf_paper = None

    llm = GoogleGenerativeAI(model="gemini-1.5-flash-001", google_api_key=GOOGLE_API_KEY)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)
    retriever_output = qa.run(user_txt)
    if len(retriever_output) > 1500:
        chunks = split_text(retriever_output, 1500)
        translated_chunks = []

        for chunk in chunks:
            translated_chunk = translator.translate(chunk, dest=user_lang).text
            translated_chunks.append(translated_chunk)

        final_output = ''.join(translated_chunks)
    else:
        final_output = translator.translate(retriever_output, dest=user_lang).text
    print(f"Translated output back to {user_lang}: {final_output}")
    return as_output, final_output, pdf_paper 
if __name__ == "__main__":
    query = input("Enter your query: ")

    if query:
        as_output, final_output, pdf_paper = query_data(query)

        print("=" * 50)
        print(f"Query: {query}")
        print("=" * 50)

        if as_output is None and final_output is None:
            print("No results found.")
        else:
            print("Atlas Vector Search Output:")
            print("-" * 30)
            print(as_output if as_output else "No output from Atlas Vector Search")

            print("Final Translated Output:")
            print("-" * 30)
            print(final_output if final_output else "No output from RAG")

            if pdf_paper:
                print("PDF Paper URL:")
                print("-" * 30)
                print(pdf_paper)

        print("=" * 50)
    else:
        print("Please enter a query.")
