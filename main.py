# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from dotenv import load_dotenv
import textwrap
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import sqlite3
from inputSql import generate_sql_from_input
import numpy as np
from filters import get_filters, get_filter_values, get_database, filter_database, get_car
from chat import reset_chat, chat_endpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

app = Flask(__name__)
CORS(app)

df = pd.read_csv('vehicles.csv')
conn = sqlite3.connect('data.db')
df.to_sql('cars', conn, index=False, if_exists='replace')
conn.close()

app.add_url_rule('/api/chat', 'reset_chat', reset_chat, methods=['DELETE'])
app.add_url_rule('/api/chat', 'chat_endpoint', chat_endpoint, methods=['POST'])

app.add_url_rule('/api/database', 'get_database', get_database, methods=['GET'])
app.add_url_rule('/api/database/filter', 'filter_database', filter_database, methods=['GET'])

app.add_url_rule('/api/database/filters', 'get_filters', get_filters, methods=['GET'])
app.add_url_rule('/api/database/filters/<filter_type>', 'get_filter_values', get_filter_values, methods=['GET'])

#filter on vin
app.add_url_rule('/api/database/vin/<vin>', 'get_car', get_car, methods=['GET'])



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)