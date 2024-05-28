# combined_app.py

import os
import json
import subprocess
import threading
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import ollama
import chromadb
import gradio as gr
import requests
import uvicorn
from groq import Groq
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize API keys and clients
api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=api_key)
groq_model = 'llama3-70b-8192'

app = FastAPI()
client = chromadb.Client()
collection = client.create_collection(name="docs")

# Initialize embedding models
embedding_model = "mxbai-embed-large"
chat_model = "llama3"

# Initialize console for debugging
console = Console()

# Function to generate embeddings
def generate_embeddings(text):
    response = ollama.embeddings(model=embedding_model, prompt=text)
    return response["embedding"]

# Example documents containing full code snippets and tips
documents = [
    {
        "id": "1",
        "content": """# Example 1: Basic FastAPI Setup
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Tip: Use @app.get("/") for simple GET requests.

# Example 2: Adding a new route
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

# Tip: Use path parameters for dynamic URLs.
""",
    },
    {
        "id": "2",
        "content": """# Example 3: Python Function with Error Handling
def divide_numbers(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    return result

# Tip: Always handle exceptions to prevent crashes.

# Example 4: Using List Comprehensions
numbers = [1, 2, 3, 4, 5]
squared_numbers = [n**2 for n in numbers]

# Tip: List comprehensions provide a concise way to create lists.
""",
    },
    {
        "id": "3",
        "content": """# Example 5: Calling External APIs
import requests

def get_weather(city):
    api_key = "your_api_key"
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(base_url)
    return response.json()

# Tip: Use the requests library for HTTP requests.

# Example 6: Using Lambda Functions
add = lambda x, y: x + y
print(add(2, 3))  # Output: 5

# Tip: Use lambda functions for small, anonymous function objects.
""",
    },
]

for doc in documents:
    embedding = generate_embeddings(doc["content"])
    collection.add(ids=[doc["id"]], embeddings=[embedding], documents=[doc["content"]])

class Message(BaseModel):
    message: str

@app.post("/chat")
async def chat(message: Message):
    user_input = message.message
    prompt = f"Using this data: {{}}. Respond to this prompt: {user_input}"

    # Generate embedding for user query
    user_embedding = generate_embeddings(user_input)
    results = collection.query(query_embeddings=[user_embedding], n_results=1)
    relevant_doc = results['documents'][0][0]

    # Generate response using chat model
    response = ollama.generate(model=chat_model, prompt=prompt.format(relevant_doc))
    initial_answer = response['response']

    # Groq API tool calling
    messages = [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses provided tools to enhance responses."
        },
        {
            "role": "user",
            "content": user_input,
        },
        {
            "role": "assistant",
            "content": initial_answer,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Execute a shell command on the server.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to run (e.g., 'ls -l')",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_file_permissions",
                "description": "Check the permissions of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to check (e.g., '/path/to/file')",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lint_code",
                "description": "Lint Python code using pylint.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to lint.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "format_code",
                "description": "Format Python code using black.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to format.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "perform_web_research",
                "description": "Perform web research using Selenium and BeautifulSoup.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for web research",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run tests on the script.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_path": {
                            "type": "string",
                            "description": "The path to the script",
                        }
                    },
                    "required": ["script_path"],
                },
            },
        }
    ]
    groq_response = groq_client.chat.completions.create(
        model=groq_model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )
    response_message = groq_response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        available_functions = {
            "run_shell_command": run_shell_command,
            "check_file_permissions": check_file_permissions,
            "lint_code": lint_code,
            "format_code": format_code,
            "perform_web_research": perform_web_research,
            "run_tests": run_tests,
        }
        messages.append(response_message)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
        final_response = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages
        )
        final_answer = final_response.choices[0].message.content
        return {"message": final_answer}

    return {"message": initial_answer}

# Function to run shell commands
def run_shell_command(command):
    try:
        result = os.popen(command).read()
        return result
    except Exception as e:
        return str(e)

# Function to check file permissions
def check_file_permissions(file_path):
    try:
        import stat
        st = os.stat(file_path)
        permissions = stat.filemode(st.st_mode)
        return permissions
    except Exception as e:
        return str(e)

# Function to lint Python code
def lint_code(code):
    try:
        with open('temp_code.py', 'w') as f:
            f.write(code)
        result = subprocess.run(['pylint', 'temp_code.py'], capture_output=True, text=True)
        os.remove('temp_code.py')
        return result.stdout
    except Exception as e:
        return str(e)

# Function to format Python code
def format_code(code):
    try:
        with open('temp_code.py', 'w') as f:
            f.write(code)
        result = subprocess.run(['black', 'temp_code.py'], capture_output=True, text=True)
        formatted_code = open('temp_code.py', 'r').read()
        os.remove('temp_code.py')
        return formatted_code
    except Exception as e:
        return str(e)

# Function to perform web research
def perform_web_research(query):
    try:
        service = ChromeService(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        driver = webdriver.Chrome(service=service, options=options)
        url = f"https://www.google.com/search?q={query}"
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
        )
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [a['href'] for a in soup.select('div.g a') if a['href'].startswith('http')]
        results = []
        for link in links[:3]:  # Limit to first 3 links for brevity
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_text = ' '.join([p.get_text() for p in page_soup.find_all('p')])
            results.append(page_text[:500])  # Limit to first 500 characters
        driver.quit()
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to run tests on a script
def run_tests(script_path):
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            return json.dumps({"result": "All tests passed."})
        else:
            return json.dumps({"error": f"Tests failed:\n{result.stderr}"})
    except Exception as e:
        return json.dumps({"error": f"Error in run_tests: {str(e)}"})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_content = contents.decode("utf-8")
    embedding = generate_embeddings(file_content)
    collection.add(ids=[file.filename], embeddings=[embedding], documents=[file_content])
    return {"filename": file.filename, "status": "uploaded and vectorized"}

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
threading.Thread(target=start_fastapi).start()

# Gradio Interface
def chat_with_bot(user_input):
    response = requests.post("http://localhost:8000/chat", json={"message": user_input})
    return response.json()["message"]

def upload_document(doc):
    files = {"file": (doc.name, str(doc), "text/plain")}
    response = requests.post("http://localhost:8000/upload", files=files)
    return response.json()

with gr.Blocks() as demo:
    gr.Markdown("# Chat with Llama Bot")
    chatbot = gr.Chatbot(height=400)
    user_input = gr.Textbox(placeholder="Type your message here...")
    upload_input = gr.File(label="Upload Document")

    def respond(message, history):
        bot_response = chat_with_bot(message)
        history.append((message, bot_response))
        return history

    def handle_upload(doc):
        upload_response = upload_document(doc)
        return f"Uploaded {upload_response['filename']} successfully!"

    user_input.submit(respond, [user_input, chatbot], chatbot)
    upload_input.upload(handle_upload, upload_input, None)

    demo.launch(share=True)
