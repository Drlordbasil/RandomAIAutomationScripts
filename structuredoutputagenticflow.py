from groq import Groq
import os
import json
import subprocess
from time import sleep
from typing import Dict, Callable, List, Any
import ollama
import chromadb
import uuid
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from rich import print
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel
from rich.text import Text

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=api_key)
MODEL = 'llama3-70b-8192'

console = Console()

def setup_driver():
    print(Panel("Setting up Chrome WebDriver...", expand=False))
    with Progress() as progress:
        task = progress.add_task("[green]Downloading ChromeDriver...", total=100)
        service = ChromeService(ChromeDriverManager().install())
        while not progress.finished:
            progress.update(task, advance=0.5)
            sleep(0.02)
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    #options.add_argument('--headless')  # Uncomment for headless mode
    print(Panel("Chrome WebDriver set up successfully!", expand=False))
    return webdriver.Chrome(service=service, options=options)

def perform_web_research(query: str) -> str:
    print(Panel(f"Performing web research for: [bold]{query}[/bold]", expand=False))
    driver = setup_driver()
    try:
        url = f"https://www.google.com/search?q={query}"
        print(f"[bold green]Navigating to:[/bold green] {url}")
        driver.get(url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
        )
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        links = [a['href'] for a in soup.select('div.g a') if a['href'].startswith('http')]
        console.rule("[bold green]Search Results[/bold green]")
        results = []
        for i, link in enumerate(links[:3], start=1):  # Limit to first 3 links for brevity
            print(f"[bold]Extracting content from result {i}:[/bold] {link}")
            driver.get(link)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_soup = BeautifulSoup(driver.page_source, 'html.parser')
            page_text = ' '.join([p.get_text() for p in page_soup.find_all('p')])
            results.append(page_text[:500])  # Limit to first 500 characters
            print(f"[green]Content extracted successfully![/green]")
        console.rule()
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        print(Panel("Closing Chrome WebDriver...", expand=False))
        driver.quit()

def run_python_script(script_content: str, input_data: str = "") -> str:
    try:
        print(Panel(f"[bold]Executing Python script:[/bold]\n{script_content}", expand=False))
        process = subprocess.Popen(
            ['python', '-c', script_content],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        with console.status("[bold green]Running Python script..."):
            stdout, stderr = process.communicate(input=input_data)
        print(Panel(f"[bold green]Python script executed successfully![/bold green]\nOutput: {stdout}\nError: {stderr}", expand=False))
        return json.dumps({"output": stdout, "error": stderr})
    except Exception as e:
        print(Panel(f"[bold red]Error executing Python script:[/bold red] {str(e)}", expand=False))
        return json.dumps({"error": str(e)})

def example_tool_function(param: str) -> str:
    print(Panel(f"[bold]Running example tool function with parameter:[/bold] {param}", expand=False))
    return json.dumps({"param": param, "result": "success"})

def run_conversation(user_prompt: str, tools: List[Dict[str, Any]], available_functions: Dict[str, Callable], conversation_history: List[Dict[str, str]], rag_collection) -> str:
    print(Panel(f"[bold]User Prompt:[/bold] {user_prompt}", expand=False))
    add_to_rag_system(user_prompt, rag_collection)
    print(Panel("[bold green]Added user prompt to RAG system.[/bold green]", expand=False))
    context = query_rag_system(user_prompt, rag_collection)
    print(Panel(f"[bold]Context retrieved from RAG:[/bold] {context}", expand=False))
    
    examples = [
        {"role": "system", "content": "You are a function calling LLM that uses the available functions to answer questions. Do not say 'TERMINATE' unless all tasks are fully complete."},
        {"role": "user", "content": "What is the weather like in New York today?"},
        {"role": "assistant", "content": "Using the weather function: The weather in New York today is sunny with a high of 75°F."},
        {"role": "user", "content": "Run this Python script: print('Hello, World!')"},
        {"role": "assistant", "content": "Running the provided Python script:\nOutput: 'Hello, World!'"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Using general knowledge: The capital of France is Paris."}
    ]

    messages = examples + conversation_history + [
        {"role": "user", "content": user_prompt},
        {"role": "system", "content": f"Context from RAG: {context}"}
    ]
    
    while True:
        try:
            print(Panel("[bold]Sending request to Groq...[/bold]", expand=False))
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096
            )
            sleep(2)
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                print(Panel("[bold]Tool calls detected in the response.[/bold]", expand=False))
                messages.append(response_message)
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)
                    function_args = json.loads(tool_call.function.arguments)
                    print(Panel(f"[bold]Executing tool: [green]{function_name}[/green] with arguments: [blue]{function_args}[/blue][/bold]", expand=False))
                    function_response = function_to_call(**function_args)
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response,
                        }
                    )
                sleep(2)
                print(Panel("[bold]Sending second request to Groq...[/bold]", expand=False))
                second_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                final_response = second_response.choices[0].message.content
            else:
                final_response = response_message.content
            
            if "TERMINATE" in final_response:
                break

            messages.append({"role": "assistant", "content": final_response})
            print(Panel(f"[bold green]Assistant's Response:[/bold green] {final_response}", expand=False))

        except Exception as e:
            return str(e)

    return final_response

def setup_rag_system() -> tuple:
    print(Panel("[bold]Setting up RAG system...[/bold]", expand=False))
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="docs")
    print(Panel("[bold green]RAG system set up successfully![/bold green]", expand=False))
    return client, collection

def add_to_rag_system(text: str, collection) -> None:
    print(Panel(f"[bold]Adding text to RAG system:[/bold] {text}", expand=False))
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    embedding = response["embedding"]
    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[embedding],
        documents=[text]
    )

def query_rag_system(prompt: str, collection) -> str:
    print(Panel(f"[bold]Querying RAG system with prompt:[/bold] {prompt}", expand=False))
    response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
    results = collection.query(
        query_embeddings=[response["embedding"]],
        n_results=1
    )
    data = results['documents'][0][0]

    print(Panel(f"[bold]Generating response using data from RAG:[/bold] {data}", expand=False))
    output = ollama.generate(
        model="llama3",
        prompt=f"Using this data: {data}. Respond to this prompt: {prompt}"
    )
    return output['response']

if __name__ == "__main__":
    user_idea = input("What do you want to do today? ")
    initial_user_prompt = f"research examples on how to use/do the users ideas {user_idea} plan out your steps and do all of them before finishing..reply 'TERMINATE' to end the conversation when. Never include TERMINATE even if you are just referencing when you will do it. mask it 'I will end it after' versus saying TERMINATE unless its fully finished."
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_python_script",
                "description": "Run a given Python script and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_content": {"type": "string", "description": "The content of the Python script to run"},
                        "input_data": {"type": "string", "description": "Input data to provide to the script", "default": ""}
                    },
                    "required": ["script_content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "example_tool_function",
                "description": "Example function for demonstrating multiple tools",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string", "description": "An example parameter"},
                    },
                    "required": ["param"],
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
                        "query": {"type": "string", "description": "The search query for web research"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    available_functions = {
        "run_python_script": run_python_script,
        "example_tool_function": example_tool_function,
        "perform_web_research": perform_web_research,
    }

    print(Panel("[bold]Setting up RAG system...[/bold]", expand=False))
    rag_client, rag_collection = setup_rag_system()
    initial_context = [
        "Python is a versatile programming language used for various applications.",
        "reply 'TERMINATE' to end the conversation.",
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data."
    ]
    for context in initial_context:
        add_to_rag_system(context, rag_collection)

    conversation_history = []

    print(Panel("[bold]Running conversation with user prompt...[/bold]", expand=False))
    result = run_conversation(initial_user_prompt, tools, available_functions, conversation_history, rag_collection)
    print(Panel(f"[bold green]Final Result:[/bold green]\n{result}", expand=False))

    conversation_history.append({"role": "user", "content": initial_user_prompt})
    conversation_history.append({"role": "assistant", "content": result})
