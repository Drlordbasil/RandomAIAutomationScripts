import json
import os
import re
import shutil
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import ollama
import chromadb
from groq import Groq

# Base class for agents
class BaseAgent:
    def __init__(self, name, groq_api_key, model, chromadb_client, workspace_dir):
        self.name = name
        self.client = Groq(api_key=groq_api_key)
        self.model = model
        self.chromadb_client = chromadb_client
        self.collection = self.chromadb_client.create_collection(name=f"{name}_docs")
        self.memory_collection = self.chromadb_client.create_collection(name=f"{name}_memory")
        self.workspace_dir = workspace_dir
        self.votes = {}

    def store_documents(self, documents):
        for i, d in enumerate(documents):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
            embedding = response["embedding"]
            self.collection.add(ids=[str(i)], embeddings=[embedding], documents=[d])

    def store_memory(self, conversation_history):
        for i, message in enumerate(conversation_history):
            response = ollama.embeddings(model="mxbai-embed-large", prompt=message)
            embedding = response["embedding"]
            self.memory_collection.add(ids=[str(i)], embeddings=[embedding], documents=[message])

    def retrieve_relevant_memory(self, query):
        response = ollama.embeddings(prompt=query, model="mxbai-embed-large")
        results = self.memory_collection.query(query_embeddings=[response["embedding"]], n_results=1)
        return results['documents'][0][0] if results['documents'] else None

    def google_search(self, query):
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            driver = webdriver.Chrome(options=chrome_options)
            search_url = f"https://www.google.com/search?q={query}"
            driver.get(search_url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
            results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd", limit=5)
            summaries = [result.text for result in results if result.text]
            return summaries if summaries else ["No relevant information found."]
        except Exception as e:
            return [f"An error occurred during the Google search: {str(e)}"]

    def click_and_scrape(self, url):
        try:
            chrome_options = Options()
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
            content = soup.find_all("p")
            paragraphs = [p.text for p in content if p.text]
            return paragraphs if paragraphs else ["No relevant information found on the page."]
        except Exception as e:
            return [f"An error occurred during the web scraping: {str(e)}"]

    def read_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                content = file.read()
            return content
        except Exception as e:
            return f"An error occurred while reading the file: {str(e)}"

    def write_file(self, file_path, content):
        try:
            full_path = os.path.join(self.workspace_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as file:
                file.write(content)
            return f"Successfully wrote to {full_path}"
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"

    def clean_code(self, file_path):
        content = self.read_file(file_path)
        cleaned_code = re.sub(r'```python\n(.*?)\n```', r'\1', content, flags=re.DOTALL)
        self.write_file(file_path, cleaned_code)
        return f"Cleaned code in {file_path}"

    def run_shell_command(self, command):
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred while running the shell command: {str(e)}"

    def run_conversation(self, user_prompt):
        conversation_history = []
        messages = [
            {"role": "system", "content": f"You are a {self.name} that performs tasks using various tools and stores information for later retrieval."},
            {"role": "user", "content": user_prompt}
        ]
        conversation_history.append(f"User: {user_prompt}")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "google_search",
                    "description": "Search Google for the given query",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "click_and_scrape",
                    "description": "Click on a link and scrape the necessary information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "The URL to scrape"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the content of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "The path of the file to read"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "The path of the file to write"},
                            "content": {"type": "string", "description": "The content to write to the file"}
                        },
                        "required": ["file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_shell_command",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "The shell command to run"}
                        },
                        "required": ["command"]
                    }
                }
            }
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, tool_choice="auto", max_tokens=4096)
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        conversation_history.append(f"{self.name}: {response_message.content}")

        if tool_calls:
            available_functions = {
                "google_search": self.google_search,
                "click_and_scrape": self.click_and_scrape,
                "read_file": self.read_file,
                "write_file": self.write_file,
                "run_shell_command": self.run_shell_command
            }
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(**function_args)
                self.store_documents(function_response)
                conversation_history.append(f"Tool call ({function_name}): {json.dumps(function_response)}")
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": json.dumps(function_response)})
            second_response = self.client.chat.completions.create(model=self.model, messages=messages)
            conversation_history.append(f"{self.name}: {second_response.choices[0].message.content}")
            self.store_memory(conversation_history)
            return second_response.choices[0].message.content
        self.store_memory(conversation_history)
        return "No tool calls were made."

class ResearchAgent(BaseAgent):
    pass

class GamePlanAgent(BaseAgent):
    def create_game_plan(self, research_results):
        messages = [
            {"role": "system", "content": "You are a project manager that creates detailed game plans based on research data."},
            {"role": "user", "content": f"Using this research data: {research_results}, create a game plan for a project."}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        return response.choices[0].message.content

class PythonExpertAgent(BaseAgent):
    def generate_code(self, task_description):
        messages = [
            {"role": "system", "content": "You are a Python expert that generates Python code for given tasks."},
            {"role": "user", "content": f"Generate Python code for the following task: {task_description}"}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        return response.choices[0].message.content

class DebuggingExpertAgent(BaseAgent):
    def debug_code(self, code):
        messages = [
            {"role": "system", "content": "You are a debugging expert that helps debug Python code."},
            {"role": "user", "content": f"Debug the following Python code: {code}"}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        return response.choices[0].message.content

class CodeReviewerAgent(BaseAgent):
    def review_code(self, code):
        messages = [
            {"role": "system", "content": "You are a code reviewer that ensures code quality and best practices."},
            {"role": "user", "content": f"Review the following Python code for quality and best practices: {code}"}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        return response.choices[0].message.content

class ManagementAgent(BaseAgent):
    def manage_project(self, game_plan):
        messages = [
            {"role": "system", "content": "You are a project manager that divides game plans into tasks and delegates them to other agents."},
            {"role": "user", "content": f"Using this game plan: {game_plan}, create tasks and delegate them to the appropriate agents."}
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        return response.choices[0].message.content

    def create_project_structure(self, project_name):
        project_path = os.path.join(self.workspace_dir, project_name)
        os.makedirs(project_path, exist_ok=True)
        os.makedirs(os.path.join(project_path, "src"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/assets"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/data"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/utils"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/core"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/functions"), exist_ok=True)
        os.makedirs(os.path.join(project_path, "src/tools"), exist_ok=True)

        self.write_file(f"{project_name}/src/main.py", "# Entry point for the RPG game\n")
        self.write_file(f"{project_name}/src/utils.py", "# Utility functions for common tasks\n")
        self.write_file(f"{project_name}/src/core.py", "# Core classes and logic for the game\n")
        self.write_file(f"{project_name}/src/functions.py", "# Functions related to game mechanics\n")
        self.write_file(f"{project_name}/src/tools.py", "# Tools for assisting with game development\n")

    def clean_workspace(self):
        try:
            shutil.rmtree(self.workspace_dir)
            os.makedirs(self.workspace_dir, exist_ok=True)
            return f"Workspace {self.workspace_dir} cleaned successfully."
        except Exception as e:
            return f"An error occurred while cleaning the workspace: {str(e)}"

class ResearchCompany:
    def __init__(self, groq_api_key, model, chromadb_client, workspace_dir):
        self.groq_api_key = groq_api_key
        self.model = model
        self.chromadb_client = chromadb_client
        self.workspace_dir = workspace_dir
        self.agents = {}

    def add_agent(self, agent_name, agent_type):
        if agent_type == 'research':
            agent = ResearchAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        elif agent_type == 'gameplan':
            agent = GamePlanAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        elif agent_type == 'python':
            agent = PythonExpertAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        elif agent_type == 'debugging':
            agent = DebuggingExpertAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        elif agent_type == 'reviewer':
            agent = CodeReviewerAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        elif agent_type == 'management':
            agent = ManagementAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        else:
            raise ValueError("Unsupported agent type.")
        self.agents[agent_name] = agent

    def remove_agent(self, agent_name):
        if agent_name in self.agents:
            del self.agents[agent_name]

    def get_agent(self, agent_name):
        return self.agents.get(agent_name)

    def run_conversation(self, agent_name, user_prompt):
        agent = self.get_agent(agent_name)
        if agent:
            return agent.run_conversation(user_prompt)
        return f"Agent {agent_name} not found."

    def retrieve_and_generate_response(self, agent_name, query_prompt):
        agent = self.get_agent(agent_name)
        if agent:
            data = agent.retrieve_relevant_memory(query_prompt)
            return agent.generate_response(data, query_prompt)
        return f"Agent {agent_name} not found."

    def create_game_plan(self, agent_name, research_results):
        agent = self.get_agent(agent_name)
        if isinstance(agent, GamePlanAgent):
            return agent.create_game_plan(research_results)
        return f"Agent {agent_name} is not a GamePlanAgent."

    def generate_code(self, agent_name, task_description):
        agent = self.get_agent(agent_name)
        if isinstance(agent, PythonExpertAgent):
            return agent.generate_code(task_description)
        return f"Agent {agent_name} is not a PythonExpertAgent."

    def debug_code(self, agent_name, code):
        agent = self.get_agent(agent_name)
        if isinstance(agent, DebuggingExpertAgent):
            return agent.debug_code(code)
        return f"Agent {agent_name} is not a DebuggingExpertAgent."

    def review_code(self, agent_name, code):
        agent = self.get_agent(agent_name)
        if isinstance(agent, CodeReviewerAgent):
            return agent.review_code(code)
        return f"Agent {agent_name} is not a CodeReviewerAgent."

    def manage_project(self, agent_name, game_plan):
        agent = self.get_agent(agent_name)
        if isinstance(agent, ManagementAgent):
            return agent.manage_project(game_plan)
        return f"Agent {agent_name} is not a ManagementAgent."

    def create_project_structure(self, agent_name, project_name):
        agent = self.get_agent(agent_name)
        if isinstance(agent, ManagementAgent):
            return agent.create_project_structure(project_name)
        return f"Agent {agent_name} is not a ManagementAgent."

    def clean_code(self, agent_name, file_path):
        agent = self.get_agent(agent_name)
        if isinstance(agent, BaseAgent):
            return agent.clean_code(file_path)
        return f"Agent {agent_name} is not a valid agent for code cleaning."

    def clean_workspace(self, agent_name):
        agent = self.get_agent(agent_name)
        if isinstance(agent, ManagementAgent):
            return agent.clean_workspace()
        return f"Agent {agent_name} is not a ManagementAgent."

# Example usage
groq_api_key = os.getenv('GROQ_API_KEY')
model = 'llama3-70b-8192'
chromadb_client = chromadb.Client()
workspace_dir = "workspace"

company = ResearchCompany(groq_api_key, model, chromadb_client, workspace_dir)
company.add_agent("ResearchAgent1", 'research')
company.add_agent("GamePlanAgent1", 'gameplan')
company.add_agent("PythonExpertAgent1", 'python')
company.add_agent("DebuggingExpertAgent1", 'debugging')
company.add_agent("CodeReviewerAgent1", 'reviewer')
company.add_agent("ManagementAgent1", 'management')

# Create project structure
project_name = "RPG_Game"
company.create_project_structure("ManagementAgent1", project_name)

# Main loop to iterate through different phases
def main_loop(company, project_name):
    # Research phase
    research_prompts = [
        "Research the best practices for creating an RPG game in Python.",
        "Research the best libraries for game development in Python.",
        "Research how to implement character creation in RPG games."
    ]
    research_results = []
    for prompt in research_prompts:
        result = company.run_conversation("ResearchAgent1", prompt)
        research_results.append(result)
        print(f"Research result: {result}")

    # Game plan creation phase
    combined_research = " ".join(research_results)
    game_plan = company.create_game_plan("GamePlanAgent1", combined_research)
    print(f"Game plan: {game_plan}")

    # Task delegation and code generation phase
    tasks = [
        {"description": "Write a Python function to handle character creation.", "file": "functions.py"},
        {"description": "Implement the main game loop.", "file": "main.py"},
        {"description": "Create utility functions for game mechanics.", "file": "utils.py"},
        {"description": "Develop core game classes and logic.", "file": "core.py"}
    ]

    for task in tasks:
        task_description = task["description"]
        file_path = f"{project_name}/src/{task['file']}"
        python_code = company.generate_code("PythonExpertAgent1", task_description)
        print(f"Generated code for {task_description}: {python_code}")

        # Write the generated code to the corresponding file
        write_response = company.run_conversation("PythonExpertAgent1", json.dumps({
            "file_path": file_path,
            "content": python_code
        }))
        print(write_response)

        # Debug the generated code
        debugged_code = company.debug_code("DebuggingExpertAgent1", python_code)
        print(f"Debugged code for {task_description}: {debugged_code}")

        # Review the debugged code
        reviewed_code = company.review_code("CodeReviewerAgent1", debugged_code)
        print(f"Reviewed code for {task_description}: {reviewed_code}")

        # Clean code if it contains markdown
        clean_response = company.clean_code("ManagementAgent1", file_path)
        print(clean_response)

    # Clean the workspace
    clean_workspace_response = company.clean_workspace("ManagementAgent1")
    print(clean_workspace_response)

    # Check for project completion
    termination_votes = [agent.vote_termination() for agent in company.agents.values()]
    if all(termination_votes):
        print("All agents agree that the project is complete.")
    else:
        print("Not all agents agree that the project is complete. Further iterations required.")

# Run the main loop
main_loop(company, project_name)
