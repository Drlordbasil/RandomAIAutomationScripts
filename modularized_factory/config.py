import os
from dotenv import load_dotenv
from groq import Groq
import tempfile
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
MODEL = 'llama3-70b-8192'
SCRIPT_PATH = tempfile.gettempdir() + '/groq_script.py'

if not os.path.exists(SCRIPT_PATH):
    with open(SCRIPT_PATH, 'w') as f:
        f.write("# Initial script content\n")
