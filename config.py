"""
ClearPath Chatbot - Configuration
Loads environment variables and defines constants.
"""

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

GROQ_API_KEY_SMALL = os.getenv("Groq_Llama_31_8b_instant", "")
GROQ_API_KEY_LARGE = os.getenv("Groq_Llama_33_70b_versatile", "")

GROQ_API_KEY = GROQ_API_KEY_LARGE or GROQ_API_KEY_SMALL

MODEL_SIMPLE = "llama-3.1-8b-instant"
MODEL_COMPLEX = "llama-3.3-70b-versatile"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "rag", "ClearPath", "clearpath_docs")
INDEX_DIR = os.path.join(BASE_DIR, "rag", "index_store")
LOG_DIR = os.path.join(BASE_DIR, "logs")
ROUTER_LOG_FILE = os.path.join(LOG_DIR, "router_logs.jsonl")

CHUNK_SIZE_WORDS = 350          
CHUNK_OVERLAP_WORDS = 50        
TOP_K_CHUNKS = 5                

INTERNAL_DOC_PREFIXES = [
    "01_Employee_Handbook",
    "02_Data_Security_Privacy_Policy",
    "03_Remote_Work_Guidelines",
    "04_Code_of_Conduct",
    "05_PTO_Leave_Policy",
    "22_Q4_2023_Team_Retrospective",
    "23_Engineering_Team_Structure",
    "24_Weekly_Standup_Notes",
    "25_Product_Roadmap_2024",
]
