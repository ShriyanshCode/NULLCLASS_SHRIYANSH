# Chatbot Projects

This repository contains two chatbot projects: **MedQuAD Q&A Chatbot** and **arXiv Qualitative Finance Chatbot**. Both projects share similar basic requirements for reusability, including the use of MongoDB Atlas and Google Gemini API.

## Basic Requirements for replication (change theses parameters in params.py)
- **MongoDB Atlas:** You need to have a MongoDB Atlas collection and account set up.
- **Google Gemini API Key:** Make sure you have a valid Google Gemini API key.
- **Create python virtual env to avoid clashing pakcage imports:**
  ```python
  py -m venv virtualenvname
  python -m venv virtualenvname

## TASK 1: MedQuAD Q&A Chatbot

### Setup and Run
1. Navigate to the `TASK1` directory:
   ```bash
   cd TASK1
2. Install required packages:
   ```bash
   pip install -r requirements.txt
2. Run chatbot:
   ```bash
   streamlit run main.py

## TASK 2: arXiv Qualitative Finance Chatbot

### Setup and Run
1. Navigate to the `TASK2` directory:
   ```bash
   cd TASK1
2. Install required packages:
   ```bash
   pip install -r requirements.txt
2. Run chatbot:
   ```bash
   streamlit run main.py
### Reusability
1. Install the required dependencies in new directory:
   ```bash
   pip install -r requirements.txt
2. Modify the params.py file to edit the database and collection names:
   ```bash
   dbName = "your_database_name"
   collectionName = "your_collection_name"
3. Load the new data into MongoDB:
   ```bash
   python load.py
4. Create a vector search index using the following parameters (since the data has been embedded with Google's embedding-001 model):
   ```bash
   {
     "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "dimensions": 768,
        "similarity": "cosine",
        "type": "knnVector"
         }
        }
      }
   }
5. Once the index is created, run the Streamlit application:
```bash
   streamlit run main.py


