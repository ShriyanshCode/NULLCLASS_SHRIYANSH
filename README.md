# Chatbot Projects

This repository contains two chatbot projects: **MedQuAD Q&A Chatbot** and **arXiv Qualitative Finance Chatbot**. Both projects share similar basic requirements for reusability, including the use of MongoDB Atlas and Google Gemini API.
Pre-processing of data has been done using kaggle in the following files:
medquad-data-preproccessing_kaggle.ipynb
arxiv-preprocessing (1).ipynb
A GPU running chatbot can be found here, in the code cell having %writefile.py:
Shriyansh_MEDquad_colab.ipynb

## Basic Requirements for replication (change theses parameters in params.py)
- **MongoDB Atlas:** You need to have a MongoDB Atlas collection and account set up.
- **Google Gemini API Key:** Make sure you have a valid Google Gemini API key.
- **Create python virtual env to avoid clashing pakcage imports:**
  ```python
  py -m venv virtualenvname
  python -m venv virtualenvname
-**Please ensure to use correct python exec:**
py filename.py or python filename.py depending on the python version you use.
This has been built using python 3.9.12.

## TASK 1: MedQuAD Q&A Chatbot

### Setup and Run
1. Navigate to the `medquad` directory:
   ```bash
   cd TASK1\medquad
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Run to enable API Keys:
   ```bash
   py params.py
   #or
   python params.py
2. Run chatbot/ GUI via Streamlit:
   ```bash
   streamlit run Chatbot_QnA.py

## TASK 2: arXiv Qualitative Finance Chatbot

### Setup and Run
1. Navigate to the `TASK2` directory:
   ```bash
   cd TASK2
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Run to enable API Keys:
   ```bash
   py params.py
   #or
   python params.py

4. Run chatbot / GUI via Streamlit:
   ```bash
   streamlit run Chatbot_main.py
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
   #or
   py load.py
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
5. Once the index is created, run:
    ```bash
   streamlit run Chatbot_QnA.py
     
