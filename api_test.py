import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
import psycopg2
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------
# LOAD ENV VARIABLES
# -----------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

if not DB_PASSWORD:
    raise ValueError("DB_PASSWORD not found in .env file")

# -----------------------
# CONFIGURE GEMINI
# -----------------------
genai.configure(api_key=GOOGLE_API_KEY)

llm = genai.GenerativeModel(
    model_name="gemini-2.5-flash"
)

# -----------------------
# DATABASE CONFIG (NOW SAFE)
# -----------------------
DB_CONFIG = {
    "host": "aws-1-ap-southeast-1.pooler.supabase.com",
    "port": 6543,
    "database": "postgres",
    "user": "postgres.mvjutxwmwcfxvthzzvif",
    "password": DB_PASSWORD,   # ← from .env now
    "sslmode": "require"
}

# -----------------------
# EMBEDDING MODEL
# -----------------------
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
model = SentenceTransformer(EMBEDDING_MODEL)

# -----------------------
# FASTAPI INIT
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# -----------------------
# EMBEDDING FUNCTION
# -----------------------
def embed(text):
    return model.encode(text).tolist()

# -----------------------
# RAG FUNCTION (UNCHANGED LOGIC)
# -----------------------
def generate_answer(query):

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    query_embedding = embed(query)

    sql = """
    SELECT text_chunk, book_name
    FROM merged_table
    ORDER BY embedding <-> %s::vector
    LIMIT 5;
    """

    cur.execute(sql, (query_embedding,))
    results = cur.fetchall()

    cur.close()
    conn.close()

    context_blocks = []
    book_names = set()

    for row in results:
        text_chunk = row[0]
        book_name = row[1]
        book_names.add(book_name)
        context_blocks.append(text_chunk)

    context = "\n\n".join(context_blocks)
    book_list = ", ".join(book_names)

    full_prompt = f"""
You are an academic textbook assistant.

You are given:
- A question from a student.
- Retrieved context from a specific textbook.
- The name of the textbook the context came from.

Your task:

1. Carefully read the retrieved context.
2. Extract the direct answer to the question strictly from the context.
3. Rewrite the answer clearly in your own words.
4. Provide additional background explanation using only the provided context.
5. Do NOT add information that is not present in the context.
6. If the answer is not clearly found, say:
   "I don't know based on the textbook."

Response Format:

Answer:
<clear answer in 3–5 sentences>

Background:
<additional explanation>

Source:
{book_list}

-----------------------------------------

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.generate_content(
        full_prompt,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 800,
        }
    )

    return response.text

# -----------------------
# API ENDPOINT
# -----------------------
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    answer = generate_answer(request.question)
    return {"answer": answer}

# -----------------------
# HEALTH CHECK
# -----------------------
@app.get("/")
def home():
    return {"message": "API is running"}
