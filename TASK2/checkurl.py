arxiv_id = 704.0394
arxiv_id_str = str(arxiv_id).strip()
if len(arxiv_id_str.split('.')[0]) < 4:  
    arxiv_id_str = '0' + arxiv_id_str
corrected_url = f"https://arxiv.org/pdf/{arxiv_id_str}.pdf"

print(corrected_url)
from pymongo.mongo_client import MongoClient

MONGO_URI = "mongodb+srv://shriyanshpatnaik2022:Ks34%40%232a@cluster0.awlceyw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(MONGO_URI)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)