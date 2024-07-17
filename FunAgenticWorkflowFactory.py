import os
import json
import subprocess
import threading
import time
from queue import Queue, Empty
from groq import Groq
from dotenv import load_dotenv
from rich.console import Console
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from textblob import TextBlob
import ollama
import chromadb
import tempfile
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)
MODEL = 'llama3-70b-8192'
EMBED_MODEL = 'mxbai-embed-large'
SCRIPT_PATH = tempfile.NamedTemporaryFile().name
RESULTS_PATH = tempfile.NamedTemporaryFile().name
CHUNK_SIZE = 1024  # Define a chunk size for the data

if not os.path.exists(SCRIPT_PATH):
    with open(SCRIPT_PATH, 'w') as f:
        f.write("# Initial script content\n")

console = Console()

# Initialize ChromaDB client and collection
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(name="docs")

def setup_driver():
    """Setup Selenium WebDriver."""
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--disable-dev-shm-usage')
    # options.add_argument('--headless')  # Run headless for no GUI
    return webdriver.Chrome(service=service, options=options)

def perform_web_research(query):
    """Perform web research using Selenium and BeautifulSoup."""
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
        return {"query": query, "results": results}
    except Exception as e:
        return {"error": str(e)}
    finally:
        driver.quit()

def run_tests(script_path):
    """Run tests on the script."""
    try:
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        if result.returncode == 0:
            return {"result": "All tests passed."}
        else:
            return {"error": f"Tests failed:\n{result.stderr}"}
    except Exception as e:
        return {"error": f"Error in run_tests: {str(e)}"}

def manage_hr(action, agent_name):
    """Manage HR tasks such as hiring or firing agents."""
    try:
        if action == 'hire':
            return {"result": f"HR tasks completed: {agent_name} has been hired."}
        elif action == 'fire':
            return {"result": f"HR tasks completed: {agent_name} has been fired."}
        else:
            return {"error": "Invalid action"}
    except Exception as e:
        return {"error": f"Error in manage_hr: {str(e)}"}

def analyze_sentiment(text):
    """Analyze the sentiment of the given text."""
    try:
        analysis = TextBlob(text)
        sentiment = analysis.sentiment
        return {"text": text, "sentiment": {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}}
    except Exception as e:
        return {"error": f"Error in analyze_sentiment: {str(e)}"}

def clean_code(script_path):
    """Clean and format the code."""
    try:
        with open(script_path, 'r') as f:
            code = f.read()
        
        # Simple cleaning: Removing extra whitespaces
        cleaned_code = "\n".join([line.strip() for line in code.split('\n') if line.strip()])

        with open(script_path, 'w') as f:
            f.write(cleaned_code)
        
        return {"result": "Code cleaned successfully."}
    except Exception as e:
        return {"error": f"Error in clean_code: {str(e)}"}

def save_results(data):
    """Save the results to a file."""
    try:
        with open(RESULTS_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        return {"result": "Results saved successfully."}
    except Exception as e:
        return {"error": f"Error in save_results: {str(e)}"}

def chunk_data(data, chunk_size=CHUNK_SIZE):
    """Chunk data into smaller pieces."""
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]

def generate_embeddings(documents):
    """Generate embeddings for a list of documents."""
    embeddings = []
    for i, doc in enumerate(documents):
        response = ollama.embeddings(model=EMBED_MODEL, prompt=doc)
        embedding = response["embedding"]
        chroma_collection.add(
            ids=[str(i)],
            embeddings=[embedding],
            documents=[doc]
        )
        embeddings.append(embedding)
    return embeddings

def retrieve_document(prompt):
    """Retrieve the most relevant document for a given prompt."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=prompt)
    query_embedding = response["embedding"]
    results = chroma_collection.query(query_embeddings=[query_embedding], n_results=1)
    return results['documents'][0][0]

def run_conversation(user_prompt):
    debug_print("Starting run_conversation")
    available_functions = {
        "perform_web_research": perform_web_research,
        "run_tests": run_tests,
        "manage_hr": manage_hr,
        "analyze_sentiment": analyze_sentiment,
        "clean_code": clean_code,
        "save_results": save_results
    }

    messages = [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses various tools to perform tasks in a software factory. "
                       "You can call tools such as 'perform_web_research', 'run_tests', 'manage_hr', 'analyze_sentiment', 'clean_code', and 'save_results' to get your tasks done."
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]
    tools = [
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
        },
        {
            "type": "function",
            "function": {
                "name": "manage_hr",
                "description": "Manage HR tasks such as hiring or firing agents.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": "The HR action to perform (hire/fire)",
                        },
                        "agent_name": {
                            "type": "string",
                            "description": "The name of the agent",
                        }
                    },
                    "required": ["action", "agent_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_sentiment",
                "description": "Analyze the sentiment of the given text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to analyze",
                        }
                    },
                    "required": ["text"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clean_code",
                "description": "Clean and format the code.",
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
        },
        {
            "type": "function",
            "function": {
                "name": "save_results",
                "description": "Save the results to a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "description": "The data to save",
                        }
                    },
                    "required": ["data"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=4096
        )
    except Exception as e:
        debug_print(f"Error in initial Groq API call: {str(e)}")
        return f"Error in run_conversation: {str(e)}"
    time.sleep(10)
    response_message = response.choices[0].message
    time.sleep(10)
    tool_calls = response_message.tool_calls

    if tool_calls:
        time.sleep(10)
        messages.append({"role": "assistant", "content": response_message.content})  # extend conversation with assistant's reply

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
                    "content": json.dumps(function_response),
                }
            )

        try:
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            second_response_content = second_response.choices[0].message.content
            time.sleep(10)
            if second_response_content.startswith('{') and second_response_content.endswith('}'):
                return json.loads(second_response_content), messages
            else:
                return second_response_content, messages
        except Exception as e:
            debug_print(f"Error in second Groq API call: {str(e)}")
            return f"Error in run_conversation: {str(e)}"

    return response_message.content if response_message else "Error: No response from initial response"

class Factory:
    def __init__(self):
        self.agents = []
        self.queue = Queue()
        self.rate_limit_reset_time = None

    def hire_agent(self, agent):
        self.agents.append(agent)
        console.print(f"[bold green]Hired {agent.name} for {agent.task}[/bold green]")

    def fire_agent(self, agent_name):
        self.agents = [agent for agent in self.agents if agent.name != agent_name]
        console.print(f"[bold red]Fired {agent_name}[/bold red]")

    def start_production(self, data):
        try:
            while not self.queue.empty():
                agent = self.queue.get()
                if self.rate_limit_reset_time and time.time() < self.rate_limit_reset_time:
                    wait_time = self.rate_limit_reset_time - time.time()
                    console.print(f"[bold yellow]Rate limit hit. Sleeping for {wait_time} seconds.[/bold yellow]")
                    time.sleep(wait_time)
                
                self.process_task(agent, data)
                
            return data
        except Exception as e:
            debug_print(f"Error in start_production: {str(e)}")
            return f"Error in start_production: {str(e)}"

    def process_task(self, agent, data):
        try:
            user_prompt = agent.get_user_prompt(data)
            debug_print(f"Agent {agent.name} performing task: {agent.task}")

            result = run_conversation(user_prompt)
            debug_print(f"Result: {result}")
            if isinstance(result, str) and result.startswith("Error"):
                raise Exception(result)

            # Generate embeddings for the data and update the memory
            if isinstance(result, dict) and "results" in result:
                documents = result["results"]
                generate_embeddings(documents)

            time.sleep(10)  # Sleep between tasks to avoid rate limits
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                self.rate_limit_reset_time = time.time() + 240  # Set reset time to 4 minutes from now
                debug_print(f"Rate limit error. Setting reset time to {self.rate_limit_reset_time}.")
            debug_print(f"Error in process_task: {str(e)}")

def debug_print(message):
    console.print(f"[bold yellow][DEBUG][/bold yellow] {message}")

class Agent:
    def __init__(self, name, task):
        self.name = name
        self.task = task
        self.chat_history = []

    def get_user_prompt(self, data):
        raise NotImplementedError("Each agent must implement the get_user_prompt method.")

class GroqAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "script_path": SCRIPT_PATH
        })

class ResearchAgent(Agent):
    def __init__(self, name, task, research_topic):
        super().__init__(name, task)
        self.research_topic = research_topic

    def get_user_prompt(self, data):
        return json.dumps({
            "query": self.research_topic
        })

class HRAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "action": data.get('hr_action', 'hire'),
            "agent_name": data.get('agent_name', 'Unnamed Agent')
        })

    def manage_staffing(self, factory):
        if len(factory.agents) < 14:
            factory.hire_agent(GroqAgent("New Groq Agent", "groq tasks"))
        elif len(factory.agents) > 14:
            factory.fire_agent(factory.agents[-1].name)

class DebuggingAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "script_path": SCRIPT_PATH
        })

class TestingAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "script_path": SCRIPT_PATH
        })

class SentimentAnalysisAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "text": data.get("text", "No text provided")
        })

class CodeCleaningAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "script_path": SCRIPT_PATH
        })

class SaveResultsAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "data": data
        })

class ManagerAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "data": data
        })

class RunFactory:
    def __init__(self):
        self.factory = Factory()
        self.hr_agent = HRAgent("HR Agent", "HR tasks")
        self.management_agent = ManagerAgent("Manager Agent", "management tasks")

    def run(self):
        self.factory.hire_agent(self.hr_agent)
        self.factory.hire_agent(self.management_agent)
        
        initial_data = {
            "script": "Initial script content",
            "hr_action": "hire",
            "agent_name": "Test Agent",
            "text": "This is a sample text for sentiment analysis."
        }

        while True:
            # The HR agent manages staffing dynamically
            self.hr_agent.manage_staffing(self.factory)

            # Dynamically assign tasks based on current needs
            tasks = [
                "utility tasks",
                #"AI research tasks",
                #"NLP research tasks",
                #"Computer Vision research tasks",
                "HR tasks",
                "debugging tasks",
                "testing tasks",
                "sentiment analysis tasks",
                "code cleaning tasks",
                "save results tasks",
                "management tasks"
            ]

            for task in tasks:
                # Dynamically create agents for each task and add to the queue
                agent_name = f"{task.replace(' ', '_')}_Agent"
                if not any(agent.name == agent_name for agent in self.factory.agents):
                    if "research" in task:
                        research_topic = task.split()[0] + " advancements in 2024"
                        self.factory.hire_agent(ResearchAgent(agent_name, task, research_topic))
                    else:
                        self.factory.hire_agent(GroqAgent(agent_name, task))

                for agent in self.factory.agents:
                    if agent.task == task:
                        self.factory.queue.put(agent)

            final_data = self.factory.start_production(initial_data)
            debug_print("Final Data: " + json.dumps(final_data, indent=2))
            time.sleep(10)

def main():
    run_factory = RunFactory()
    factory_thread = threading.Thread(target=run_factory.run)
    factory_thread.start()
    factory_thread.join()

if __name__ == "__main__":
    main()
