# CareBot-AI
CareBot AI is an end-to-end medical chatbot built with LLMs and Flask. It answers healthcare-related queries by retrieving information from validated medical data sources, focusing on symptoms, diagnoses, and treatments. 
Technologies Used: Flask, Gemini API, LangChain, Pinecone, Hugging Face

Output:
![image](https://github.com/user-attachments/assets/d023a343-9a5b-4d1d-9c64-449c7ad342c8)
![image](https://github.com/user-attachments/assets/c99ccb02-01e9-4c3b-8c52-d89baf1c89c6)

How to run?
STEP 01- Create a conda environment after opening the repository
conda create -n carebot python=3.10 -y
conda activate carebot

STEP 02- install the requirements
pip install -r requirements.txt

STEP 03- Create API in:
pinecone.io
![image](https://github.com/user-attachments/assets/2fc6a1af-15be-4ea4-abb1-66a56f179ef1)
![image](https://github.com/user-attachments/assets/a8e980c5-3a0c-4cf4-865f-a3a02019279c)

Gemini API Key
https://aistudio.google.com/app/apikey
![image](https://github.com/user-attachments/assets/b902244e-be06-4851-b64d-d2a5e4a388fc)

Create a .env file in the root directory and add your Pinecone & geminiai credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# run the following command to store embeddings to pinecone
python store_index.py

# Finally run the following command
python app.py
Now,

open up localhost:
Techstack Used:
Python
LangChain
Flask
Gemini API
Pinecone
