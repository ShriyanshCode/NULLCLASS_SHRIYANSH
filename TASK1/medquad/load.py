import getpass
import os
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from pymongo import MongoClient
from dotenv import load_dotenv
from params import MONGO_URI, GOOGLE_API_KEY, HF_TOKEN

load_dotenv()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

client = MongoClient(MONGO_URI)

dbName = "medquad_1k"
collectionName = "qna_collections"
collection = client[dbName][collectionName]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

dir_path='1k_chunks'
all_documents = []

for file_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file_name)
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            data = []
            for entry in json_data.get('data', []):
                question = entry.get('question')
                answer = entry.get('answer')
                if question and answer:
                    # Create a document with consistent "question" and "answer" fields
                    document = {
                        "question": question,
                        "answer": answer
                    }
                    data.append(document)
            
            # Prepare documents for vectorization
            documents = [
                Document(
                    page_content=f"Question: {doc['question']}\nAnswer: {doc['answer']}",
                    metadata={"question": doc["question"], "answer": doc["answer"]}
                ) for doc in data
            ]
            
            all_documents.extend(documents)

            collection.insert_many(data)

vectorStore = MongoDBAtlasVectorSearch.from_documents(
    all_documents,
    embeddings,
    collection=collection
)

print("Documents have been successfully processed and stored in MongoDB.")
