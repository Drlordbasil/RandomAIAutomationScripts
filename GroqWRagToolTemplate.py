import os
import json
import subprocess
from time import sleep
from typing import Dict, Callable, List, Any
import ollama
import chromadb
import uuid
import logging
from groq import Groq

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = 'llama3-70b-8192'

def run_python_script(script_content: str, input_data: str = "") -> str:
    """
    Run a given Python script and return its output.
    
    Args:
        script_content (str): The content of the Python script to run.
        input_data (str, optional): Input data to provide to the script (if needed). Defaults to "".
        
    Returns:
        str: A JSON string containing the script output and any errors encountered.
    """
    try:
        logging.info(f"Running script: {script_content}")
        process = subprocess.Popen(
            ['python', '-c', script_content],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=input_data)
        result = {"output": stdout, "error": stderr}
        logging.info(f"Script output: {stdout}")
        logging.error(f"Script error: {stderr}")
        return json.dumps(result)
    except Exception as e:
        logging.error(f"Error running script: {e}")
        return json.dumps({"error": str(e)})

def example_tool_function(param: str) -> str:
    """
    Example tool function placeholder. Replace this with actual implementation.
    
    Args:
        param (str): Example parameter.
        
    Returns:
        str: A JSON string with the result.
    """
    logging.info(f"Executing example tool function with param: {param}")
    return json.dumps({"param": param, "result": "success"})

def task_complete() -> str:
    """
    Signal that the task is complete and end the loop.
    
    Returns:
        str: A JSON string indicating task completion.
    """
    logging.info("Task complete signal received.")
    return json.dumps({"status": "task_complete"})

def get_user_feedback(prompt: str) -> str:
    """
    Get feedback or input from the user.
    
    Args:
        prompt (str): The prompt to ask the user.
        
    Returns:
        str: A JSON string containing the user's feedback.
    """
    logging.info(f"Asking user for feedback: {prompt}")
    user_input = input(f"{prompt}\nYour response: ")
    return json.dumps({"user_input": user_input})

def run_conversation(user_prompt: str, tools: List[Dict[str, Any]], available_functions: Dict[str, Callable], conversation_history: List[Dict[str, str]]) -> str:
    """
    Run a conversation using the defined tools and functions.
    
    Args:
        user_prompt (str): The user prompt to initiate the conversation.
        tools (List[Dict[str, Any]]): A list of tool definitions.
        available_functions (Dict[str, Callable]): A dictionary mapping function names to function implementations.
        conversation_history (List[Dict[str, str]]): A list of previous conversation messages to maintain context.
        
    Returns:
        str: The final response from the conversation.
    """
    messages = conversation_history + [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses the available functions to answer questions. Include the details provided by the functions in your responses."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    
    while True:
        try:
            logging.info("Starting conversation with user prompt")
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=4096
            )
            sleep(2)  # Sleep timer between API calls
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls

            if tool_calls:
                messages.append({"role": "assistant", "content": response_message.content})
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_to_call = available_functions.get(function_name)
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
                    # Check if the task is complete
                    if function_name == "task_complete" and json.loads(function_response)["status"] == "task_complete":
                        return "Task completed successfully."

                    # Automatically handle errors
                    if function_name == "run_python_script":
                        script_output = json.loads(function_response)
                        print(f"Script output: {script_output['output']}")
                        print(f"Script error: {script_output['error']}")
                        if script_output["error"]:
                            error_response = client.chat.completions.create(
                                model=MODEL,
                                messages=messages + [{"role": "user", "content": script_output['error']}]
                            )
                            error_fix_message = error_response.choices[0].message
                            messages.append({"role": "assistant", "content": error_fix_message.content})
                            print(f"Error fix response: {error_fix_message.content}")
                            continue  # Retry with the new information

                sleep(2)  # Sleep timer between API calls
                second_response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                response_message = second_response.choices[0].message
                if "user_input" in response_message.content:
                    user_feedback = json.loads(response_message.content)["user_input"]
                    messages.append({"role": "user", "content": user_feedback})
            else:
                return response_message.content

        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return str(e)

def setup_rag_system() -> tuple:
    """
    Set up the RAG (Retrieval-Augmented Generation) system using Ollama and ChromaDB.
    
    Returns:
        tuple: A tuple containing the RAG system and the collection client.
    """
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="docs")
    logging.info("RAG system setup complete")
    return client, collection

def add_to_rag_system(text: str, collection) -> None:
    """
    Add a document to the RAG system.
    
    Args:
        text (str): The document text to add.
        collection: The ChromaDB collection client.
    """
    try:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding = response["embedding"]
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text]
        )
        logging.info(f"Added document to RAG system: {text}")
    except Exception as e:
        logging.error(f"Error adding document to RAG system: {e}")

def query_rag_system(prompt: str, collection) -> str:
    """
    Query the RAG system to retrieve the most relevant document and generate an answer.
    
    Args:
        prompt (str): The user's query prompt.
        collection: The ChromaDB collection client.
        
    Returns:
        str: The generated response based on the retrieved document and prompt.
    """
    try:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=prompt)
        results = collection.query(
            query_embeddings=[response["embedding"]],
            n_results=1
        )
        data = results['documents'][0][0]

        detailed_prompt = f"Using the retrieved document data: {data}. Generate a detailed response for the following prompt: {prompt}"
        output = ollama.generate(
            model="llama3",
            prompt=detailed_prompt
        )
        logging.info("Query to RAG system successful")
        return output['response']
    except Exception as e:
        logging.error(f"Error querying RAG system: {e}")
        return str(e)

if __name__ == "__main__":
    user_prompt = "create snake in python using kivy and then use run_python_script function call to run the code"
    
    # Define the tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_python_script",
                "description": "Run a given Python script and return its output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "script_content": {
                            "type": "string",
                            "description": "The content of the Python script to run",
                        },
                        "input_data": {
                            "type": "string",
                            "description": "Input data to provide to the script (if needed)",
                            "default": ""
                        }
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
                        "param": {
                            "type": "string",
                            "description": "An example parameter",
                        }
                    },
                    "required": ["param"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "task_complete",
                "description": "Signal that the task is complete and end the loop",
                "parameters": {
                    "type": "object",
                    "properties": {}
                },
                "required": [],
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_feedback",
                "description": "Get feedback or input from the user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The prompt to ask the user",
                        }
                    },
                    "required": ["prompt"],
                },
            },
        }
    ]

    # Map function names to function implementations
    available_functions = {
        "run_python_script": run_python_script,
        "example_tool_function": example_tool_function,
        "task_complete": task_complete,
        "get_user_feedback": get_user_feedback,
    }

    # Set up the RAG system
    rag_client, rag_collection = setup_rag_system()

    # Add some initial context to the RAG system
    initial_context = [
        """
        **Using the `run_python_script` function:**
        
        The `run_python_script` function takes a Python script as a string and optionally input data, runs the script, and returns the output and any errors.
        
        **Parameters:**
        - `script_content` (str): The content of the Python script to run.
        - `input_data` (str, optional): Input data to provide to the script. Defaults to "".
        
        **Returns:**
        - str: A JSON string containing the script output and any errors encountered.
        
        **Example Usage:**
        
        ```
        result = run_python_script(\"print('Hello, World!')\")
        print(result)
        ```
        
        **Expected Output:**
        ```
        {"output": "Hello, World!\\n", "error": ""}
        ```
        """,
        """
        **Using the `example_tool_function` function:**
        
        The `example_tool_function` is a placeholder function that demonstrates how to define and use a tool. It takes a single parameter and returns a JSON string with the parameter and a success message.
        
        **Parameters:**
        - `param` (str): Example parameter.
        
        **Returns:**
        - str: A JSON string with the result.
        
        **Example Usage:**
        
        ```
        result = example_tool_function("example")
        print(result)
        ```
        
        **Expected Output:**
        ```
        {"param": "example", "result": "success"}
        ```
        """,
        "Python is a versatile programming language used for various applications.",
        "Machine learning is a field of artificial intelligence that uses statistical techniques to give computers the ability to learn from data.",
        "Common Python errors include syntax errors, name errors, type errors, and index errors. Always check your code for typos and proper indentation.",
        "Scikit-learn is a popular machine learning library in Python. It provides simple and efficient tools for data mining and data analysis."
    ]
    for context in initial_context:
        add_to_rag_system(context, rag_collection)

    # Dynamically generate the RAG prompt based on the user prompt
    rag_prompt = user_prompt.split(':', 1)[-1].strip()
    rag_response = query_rag_system(rag_prompt, rag_collection)
    print(f"RAG Response: {rag_response}")

    # Initialize conversation history
    conversation_history = []

    # Run the conversation tool
    result = run_conversation(user_prompt, tools, available_functions, conversation_history)
    print(result)

    # Update conversation history
    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history.append({"role": "assistant", "content": result})

    # Continue with the next iteration
    while True:
        user_prompt = input("Please enter the next user prompt: ")
        conversation_history.append({"role": "user", "content": user_prompt})
        result = run_conversation(user_prompt, tools, available_functions, conversation_history)
        print(result)
        conversation_history.append({"role": "assistant", "content": result})
