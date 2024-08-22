import getpass
import os
import json

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document

from pymongo import MongoClient
from dotenv import load_dotenv
from params import MONGO_URI, GOOGLE_API_KEY, HF_TOKEN

load_dotenv()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

client = MongoClient(MONGO_URI)

dbName = "arxiv_papers"
collectionName = "qfin"
collection = client[dbName][collectionName]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

dir_path = 'fin_filtered_data_chunks1'
all_documents = []

for file_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file_name)
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            data = []
            for entry in json_data:
                document = {
                    "paper_id": entry.get('paper_id'),
                    "authors": entry.get('authors'),
                    "title": entry.get('title'),
                    "comments": entry.get('comments'),
                    "journal-ref": entry.get('journal-ref'),
                    "doi": entry.get('doi'),
                    "license": entry.get('license'),
                    "abstract": entry.get('abstract')
                }
                data.append(document)

            documents = [
                Document(
                    page_content=(
                        f"Paper ID: {doc['paper_id']}\n"
                        f"Title: {doc['title']}\n"
                        f"Authors: {doc['authors']}\n"
                        f"Abstract: {doc['abstract']}\n"
                        f"Comments: {doc['comments']}\n"
                        f"Journal Reference: {doc['journal-ref']}\n"
                        f"DOI: {doc['doi']}\n"
                        f"License: {doc['license']}"
                    ),
                    metadata={**doc}  # Include all metadata
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
