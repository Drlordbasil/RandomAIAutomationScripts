import os
import json
import subprocess
from groq import Groq
from typing import Any, Dict

# Initialize the Groq client with the API key
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables")

client = Groq(api_key=api_key)
MODEL = 'llama3-70b-8192'

def get_file_content(file_path: str, encoding: str = 'utf-8') -> str:
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            content = file.read()
        return json.dumps({"file_path": file_path, "content": content})
    except UnicodeDecodeError:
        return json.dumps({"file_path": file_path, "error": f"UnicodeDecodeError: Unable to decode file with {encoding} encoding"})
    except Exception as e:
        return json.dumps({"file_path": file_path, "error": str(e)})

def search_text_in_files(directory: str, text: str, encoding: str = 'utf-8') -> str:
    matches = []
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        if text in f.read():
                            matches.append(file_path)
                except UnicodeDecodeError:
                    pass
        return json.dumps({"directory": directory, "text": text, "matches": matches})
    except Exception as e:
        return json.dumps({"directory": directory, "text": text, "error": str(e)})

def run_shell_command(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return json.dumps({"command": command, "output": result.stdout, "error": result.stderr})
    except Exception as e:
        return json.dumps({"command": command, "error": str(e)})

def ask_user(question: str) -> str:
    """Function to ask the user a question via the terminal"""
    answer = input(question)
    return json.dumps({"question": question, "answer": answer})

def end_conversation() -> str:
    """Function to end the conversation"""
    return json.dumps({"status": "END_CONVO", "message": "Conversation ended by the agent."})

def run_conversation(user_prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": ("You are a function calling LLM that uses the data extracted "
                        "from various local software engineering tools to answer questions or perform tasks. "
                        "The available tools are: get_file_content (to read the content of a specified file), "
                        "search_text_in_files (to search for a specific text pattern in files within a directory), "
                        "run_shell_command (to execute a shell command and return its output), "
                        "ask_user (to ask the user a question via the terminal), "
                        "and end_conversation (to end the conversation).")
        },
        {"role": "user", "content": user_prompt}
    ]
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_file_content",
                "description": "Read and return the content of a specified file. Provide the file path and optionally the encoding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to be read.",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "The encoding to use for reading the file. Default is 'utf-8'.",
                            "default": "utf-8"
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_text_in_files",
                "description": "Search for a specific text pattern in files within a directory. Provide the directory path, the text to search for, and optionally the encoding.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": "The directory to search within.",
                        },
                        "text": {
                            "type": "string",
                            "description": "The text pattern to search for.",
                        },
                        "encoding": {
                            "type": "string",
                            "description": "The encoding to use for reading the files. Default is 'utf-8'.",
                            "default": "utf-8"
                        }
                    },
                    "required": ["directory", "text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell_command",
                "description": "Execute a shell command and return its output. Provide the command to execute.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_user",
                "description": "Ask the user a question via the terminal. Provide the question to ask.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the user.",
                        }
                    },
                    "required": ["question"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "end_conversation",
                "description": "End the conversation.",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "required": [],
            }
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
    tool_calls = getattr(response_message, 'tool_calls', [])
    all_responses = []
    
    while tool_calls:
        available_functions = {
            "get_file_content": get_file_content,
            "search_text_in_files": search_text_in_files,
            "run_shell_command": run_shell_command,
            "ask_user": ask_user,
            "end_conversation": end_conversation
        }
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            all_responses.append(function_response)
            
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )
            
            if function_name == "end_conversation":
                summary = "Summary of actions taken:\n" + "\n".join(all_responses)
                feedback = get_feedback(response_message.content)
                return f"{summary}\n\nFeedback:\n{feedback}"
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096
        )
        
        response_message = response.choices[0].message
        tool_calls = getattr(response_message, 'tool_calls', [])
    
    summary = "Summary of actions taken:\n" + "\n".join(all_responses)
    feedback = get_feedback(response_message.content)
    
    return f"{summary}\n\nFeedback:\n{feedback}"

def get_feedback(response_content: str) -> str:
    feedback_system_message = {
        "role": "system",
        "content": ("You are Anthony Snider, an expert software engineer. "
                    "Your task is to review the code provided and give feedback on its accuracy and efficiency.")
    }
    feedback_messages = [
        feedback_system_message,
        {"role": "user", "content": response_content}
    ]
    
    feedback_response = client.chat.completions.create(
        model=MODEL,
        messages=feedback_messages,
        max_tokens=4096
    )
    
    return feedback_response.choices[0].message.content

# Example user prompt
user_prompt = "Search for the text 'def run_conversation' in the current directory."
print(run_conversation(user_prompt))
