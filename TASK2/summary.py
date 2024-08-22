import google.generativeai as genai
import os
from params import GOOGLE_API_KEY

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')
import requests

pdf_url = "https://arxiv.org/pdf/0704.0589"
pdf_path = "gemini.pdf"

response = requests.get(pdf_url)
with open(pdf_path, 'wb') as f:
    f.write(response.content)

print(f"Downloaded file from {pdf_url} and saved as {pdf_path}")

sample_file = genai.upload_file(path=pdf_path, display_name="Gemini 1.5 PDF")
print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")

file = genai.get_file(name=sample_file.name)
print(f"Retrieved file '{file.display_name}' as: {file.uri}")

model = genai.GenerativeModel(model_name="gemini-1.5-flash")
response = model.generate_content([sample_file, "Can you summarize this document as a bulleted list, under 200 words?"])

print(response.text)