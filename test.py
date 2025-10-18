from mistralai import Mistral
import os

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
print("✅ Attributs disponibles :", dir(client))
