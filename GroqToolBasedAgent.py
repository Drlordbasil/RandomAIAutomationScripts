import os
import json
import subprocess
import threading
from dotenv import load_dotenv
from groq import Groq
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
from turtle import Turtle, Screen
from youtube_transcript_api import YouTubeTranscriptApi
import requests

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
client = Groq(api_key=api_key)
MODEL = 'llama3-70b-8192'

# Set up Selenium WebDriver with Chrome
def setup_driver():
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-running-insecure-content')
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# Function for web research using Selenium and BeautifulSoup
def perform_web_research(query):
    """
    This function performs web research by querying Google Search and retrieving the content
    of the first three links found. It uses Selenium WebDriver to automate the web browser
    and BeautifulSoup to parse the HTML content of the pages.

    Parameters:
    query (str): The search query for web research.

    Returns:
    json: A JSON object containing the search query and the first 500 characters of the content
          from the first three links.
    """
    driver = setup_driver()
    try:
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
        return json.dumps({"query": query, "results": results})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        driver.quit()

# Function to save a project to a file
def save_project(content, filename):
    """
    This function saves a given content string to a specified file.

    Parameters:
    content (str): The content to be saved in the file.
    filename (str): The name of the file where the content will be saved.

    Returns:
    json: A JSON object indicating the status of the save operation.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        return json.dumps({"status": "success", "filename": filename})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to check the file contents for quality and errors
def check_file_contents(filename):
    """
    This function checks the contents of a specified file for quality and errors. It ensures
    that the file has a sufficient number of lines of code and performs basic validation.

    Parameters:
    filename (str): The name of the file to be checked.

    Returns:
    json: A JSON object indicating the status of the check and feedback on the file's content.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        lines = content.strip().split('\n')
        if len(lines) < 10:
            return json.dumps({"status": "fail", "feedback": "The file has less than 10 lines of code. Add more content."})
        return json.dumps({"status": "success", "feedback": "The file content is sufficient."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to test and run Python code
def test_code(code):
    """
    This function tests and runs a given Python code snippet by executing it in a subprocess.
    It captures and returns the output or errors generated during the execution.

    Parameters:
    code (str): The Python code to be tested and run.

    Returns:
    json: A JSON object indicating the status of the test and the output or errors.
    """
    try:
        completed_process = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=30
        )
        if completed_process.returncode != 0:
            return json.dumps({"status": "fail", "output": completed_process.stderr})
        return json.dumps({"status": "success", "output": completed_process.stdout})
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "fail", "output": "Code execution timed out."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to run pylint on a file
def run_pylint(filename):
    """
    This function runs pylint on a specified Python file to check for code quality and errors.
    It captures and returns the pylint output, including any linting errors or warnings.

    Parameters:
    filename (str): The name of the Python file to be linted.

    Returns:
    json: A JSON object indicating the status of the pylint run and the output.
    """
    try:
        completed_process = subprocess.run(
            ["pylint", filename],
            capture_output=True,
            text=True,
            timeout=30
        )
        if completed_process.returncode != 0:
            return json.dumps({"status": "fail", "output": completed_process.stderr})
        return json.dumps({"status": "success", "output": completed_process.stdout})
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "fail", "output": "Pylint execution timed out."})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to visualize code execution
def visualize_code_execution(code):
    """
    This function visualizes the execution of a given Python code snippet using the turtle graphics module.
    It runs the code in an execution context that includes a turtle object for drawing.

    Parameters:
    code (str): The Python code to be visualized.

    Returns:
    None
    """
    screen = Screen()
    turtle = Turtle()

    exec_locals = {'turtle': turtle}

    exec(code, {}, exec_locals)

    screen.mainloop()

# Function to get YouTube transcript
def get_youtube_transcript(video_id):
    """
    This function retrieves the transcript of a specified YouTube video using the YouTube Transcript API.
    The video ID is required to fetch the transcript.

    Parameters:
    video_id (str): The ID of the YouTube video.

    Returns:
    json: A JSON object indicating the status of the request and the transcript content.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return json.dumps({"status": "success", "transcript": transcript})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to search GitHub repositories
def search_github_repositories(query):
    """
    This function searches GitHub repositories based on a given query string. It uses the GitHub Search API
    to retrieve information about repositories that match the query.

    Parameters:
    query (str): The search query for GitHub repositories.

    Returns:
    json: A JSON object indicating the status of the search and the results.
    """
    try:
        response = requests.get(f"https://api.github.com/search/repositories?q={query}")
        if response.status_code == 200:
            return json.dumps({"status": "success", "results": response.json()})
        return json.dumps({"error": f"GitHub API request failed with status code {response.status_code}"})
    except Exception as e:
        return json.dumps({"error": str(e)})

# Function to execute a Windows command
def execute_windows_command(command):
    """
    This function executes a given command in the Windows Command Prompt and captures the output or errors generated.

    Parameters:
    command (str): The Windows command to be executed.

    Returns:
    json: A JSON object indicating the status of the command execution and the output or errors.
    """
    try:
        completed_process = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30
        )
        if completed_process.returncode != 0:
            return json.dumps({"status": "fail", "output": completed_process.stderr})
        return json.dumps({"status": "success", "output": completed_process.stdout})
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "fail", "output": "Command execution timed out."})
    except Exception as e:
        return json.dumps({"status": "fail", "output": str(e)})

# Function to run conversation with AI and call functions
def run_conversation(user_input, context):
    messages = context + [
        {
            "role": "user",
            "content": user_input,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "perform_web_research",
                "description": "This function performs web research by querying Google Search and retrieving the content of the first three links found. It uses Selenium WebDriver to automate the web browser and BeautifulSoup to parse the HTML content of the pages. The function should be used when comprehensive web-based information is needed based on a specific query. Parameters: query (str): The search query for web research.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for web research.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "save_project",
                "description": "This function saves a given content string to a specified file. It should be used to persist data or code to a file on disk for future reference or execution. Parameters: content (str): The content to be saved in the file. filename (str): The name of the file where the content will be saved.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The content of the project.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "The filename to save the project.",
                        }
                    },
                    "required": ["content", "filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_file_contents",
                "description": "This function checks the contents of a specified file for quality and errors. It ensures that the file has a sufficient number of lines of code and performs basic validation. It should be used to validate the initial state of a file before further processing or testing. Parameters: filename (str): The name of the file to be checked.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename to check.",
                        }
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "test_code",
                "description": "This function tests and runs a given Python code snippet by executing it in a subprocess. It captures and returns the output or errors generated during the execution. This function should be used to validate Python code snippets for correctness and functionality. Parameters: code (str): The Python code to be tested and run.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to run and test.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_pylint",
                "description": "This function runs pylint on a specified Python file to check for code quality and errors. It captures and returns the pylint output, including any linting errors or warnings. Use this function to ensure the Python code adheres to coding standards and best practices. Parameters: filename (str): The name of the Python file to be linted.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The filename to lint.",
                        }
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "visualize_code_execution",
                "description": "This function visualizes the execution of a given Python code snippet using the turtle graphics module. It runs the code in an execution context that includes a turtle object for drawing. Use this function to create visual representations of code logic, especially for educational or debugging purposes. Parameters: code (str): The Python code to be visualized.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to visualize.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_youtube_transcript",
                "description": "This function retrieves the transcript of a specified YouTube video using the YouTube Transcript API. The video ID is required to fetch the transcript. Use this function to obtain the text content of a YouTube video for analysis, summarization, or other purposes. Parameters: video_id (str): The ID of the YouTube video.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "video_id": {
                            "type": "string",
                            "description": "The ID of the YouTube video.",
                        }
                    },
                    "required": ["video_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_github_repositories",
                "description": "This function searches GitHub repositories based on a given query string. It uses the GitHub Search API to retrieve information about repositories that match the query. Use this function to find relevant GitHub repositories for a given topic, technology, or project. Parameters: query (str): The search query for GitHub repositories.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for GitHub repositories.",
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_windows_command",
                "description": "This function executes a given command in the Windows Command Prompt and captures the output or errors generated. Use this function to run system-level commands on a Windows machine. Parameters: command (str): The Windows command to be executed.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The Windows command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=4096
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    # Check if the model wanted to call a function
    if tool_calls:
        available_functions = {
            "perform_web_research": perform_web_research,
            "save_project": save_project,
            "check_file_contents": check_file_contents,
            "test_code": test_code,
            "run_pylint": run_pylint,
            "visualize_code_execution": visualize_code_execution,
            "get_youtube_transcript": get_youtube_transcript,
            "search_github_repositories": search_github_repositories,
            "execute_windows_command": execute_windows_command,
        }
        messages.append(response_message)  # extend conversation with assistant's reply

        # Call the function
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
            )  # extend conversation with function response
        
        # Get a new response from the model where it can see the function response
        second_response = client.chat.completions.create(
            model=MODEL,
            messages=messages
        )
        return second_response.choices[0].message.content, messages

    return response_message.content, messages

# GUI code
def start_agent():
    user_input = prompt_entry.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showerror("Error", "Please enter a prompt.")
        return
    
    output_text.delete("1.0", tk.END)
    
    def run(user_input):
        context = [
            {
                "role": "system",
                "content": ("You are a highly skilled AI programmer and web researcher. You can call functions to perform web research, write code, save projects, check file contents for quality and errors, run pylint, and visualize Python code execution. "
                            "You must wait for the user to request tests explicitly. Continue working until the project is complete and the response includes 'TERMINATE'. "
                            "Break down tasks into smaller subtasks and automate the workflow by planning ahead. Always explain your plan before executing the tasks.")
            }
        ]
        try:
            response, updated_context = run_conversation(user_input, context)
            output_text.insert(tk.END, response + "\n")
            
            # Continuously converse until the agent decides to terminate
            while "TERMINATE" not in response:
                output_text.insert(tk.END, response + "\n")
                user_input = prompt_entry.get("1.0", tk.END).strip()
                response, updated_context = run_conversation(user_input, updated_context)
        except Exception as e:
            output_text.insert(tk.END, f"Error: {str(e)}\n")
    
    threading.Thread(target=run, args=(user_input,)).start()

def save_output():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(output_text.get("1.0", tk.END))
            messagebox.showinfo("Success", "Output saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save output: {str(e)}")

def clear_output():
    if messagebox.askyesno("Confirm", "Are you sure you want to clear the output?"):
        output_text.delete("1.0", tk.END)

# Initialize the main window
root = tk.Tk()
root.title("AI Programmer and Web Researcher")
root.geometry("800x600")

# Add styles
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.configure("TLabel", font=("Helvetica", 12))
style.configure("TEntry", font=("Helvetica", 12))
style.configure("TText", font=("Helvetica", 12))

# Create and place widgets
frame = ttk.Frame(root, padding="10")
frame.pack(fill=tk.BOTH, expand=True)

ttk.Label(frame, text="Enter your prompt:").pack(pady=5)
prompt_entry = scrolledtext.ScrolledText(frame, width=80, height=10, font=("Helvetica", 12))
prompt_entry.pack(pady=5)

button_frame = ttk.Frame(frame)
button_frame.pack(pady=10)

start_button = ttk.Button(button_frame, text="Start", command=start_agent)
start_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(button_frame, text="Save Output", command=save_output)
save_button.pack(side=tk.LEFT, padx=5)

clear_button = ttk.Button(button_frame, text="Clear Output", command=clear_output)
clear_button.pack(side=tk.LEFT, padx=5)

ttk.Label(frame, text="Output:").pack(pady=5)
output_text = scrolledtext.ScrolledText(frame, width=80, height=20, font=("Helvetica", 12))
output_text.pack(pady=5)

# Run the application
root.mainloop()
