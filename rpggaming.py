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
from PIL import Image
import pytesseract

# Base class for agents
class BaseAgent:
    def __init__(self, name, groq_api_key, model, chromadb_client, workspace_dir):
        self.name, self.groq_api_key, self.client, self.model, self.chromadb_client, self.workspace_dir = name, groq_api_key, Groq(api_key=groq_api_key), model, chromadb_client, workspace_dir
        self.memory_collection = self.chromadb_client.create_collection(name=f"{name}_memory")
        self.agents = {}

    def store_memory(self, conversation_history):
        for i, message in enumerate(conversation_history):
            self.memory_collection.add(ids=[str(i)], embeddings=[ollama.embeddings(model="mxbai-embed-large", prompt=message)["embedding"]], documents=[message])

    def retrieve_relevant_memory(self, query):
        results = self.memory_collection.query(query_embeddings=[ollama.embeddings(prompt=query, model="mxbai-embed-large")["embedding"]], n_results=3)
        if results['documents']:
            return [doc[0] for doc in results['documents'] if doc]
        return []

    def write_file(self, file_path, content):
        try:
            full_path = os.path.join(self.workspace_dir, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as file:
                file.write(content)
            return f"Successfully wrote to {full_path}"
        except Exception as e:
            return f"An error occurred while writing to the file: {str(e)}"

    def run_conversation(self, user_prompt, include_memory=True):
        conversation_history, messages = [f"User: {user_prompt}"], [{"role": "system", "content": f"You are a {self.name} that performs various tasks and stores information for later retrieval."}, {"role": "user", "content": user_prompt}]
        if include_memory:
            relevant_memory = self.retrieve_relevant_memory(user_prompt)
            if relevant_memory:
                memory_content = "\n".join(relevant_memory)
                messages.append({"role": "system", "content": f"Relevant memory:\n{memory_content}"})
        response = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=4096)
        conversation_history.append(f"{self.name}: {response.choices[0].message.content}")
        self.store_memory(conversation_history)
        return conversation_history[-1].split(": ", 1)[1]

    def create_project_structure(self, project_name):
        project_path = os.path.join(self.workspace_dir, project_name)
        for subdir in ["src/assets", "src/data", "src/utils", "src/core", "src/functions", "src/tools"]:
            os.makedirs(os.path.join(project_path, subdir), exist_ok=True)
        for file, content in [("src/main.py", "# Entry point for the RPG game\n"), ("src/utils.py", "# Utility functions for common tasks\n"), ("src/core.py", "# Core classes and logic for the game\n"), ("src/functions.py", "# Functions related to game mechanics\n"), ("src/tools.py", "# Tools for assisting with game development\n")]:
            self.write_file(f"{project_name}/{file}", content)

    def capture_screenshot(self, url, screenshot_path):
        try:
            chrome_options = Options()
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            time.sleep(2)
            driver.save_screenshot(screenshot_path)
            driver.quit()
            return True
        except Exception as e:
            print(f"An error occurred during screenshot capture: {str(e)}")
            return False

    def extract_text_from_image(self, image_path):
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"An error occurred during text extraction: {str(e)}")
            return ""

    def capture_and_describe(self, url):
        screenshot_path = os.path.join(self.workspace_dir, "screenshot.png")
        if self.capture_screenshot(url, screenshot_path):
            description = self.extract_text_from_image(screenshot_path)
            return description
        return "Failed to capture screenshot and extract text."

    def google_search(self, query):
        try:
            chrome_options = Options()
            driver = webdriver.Chrome(options=chrome_options)
            search_url = f"https://www.google.com/search?q={query}"
            driver.get(search_url)
            time.sleep(2)
            soup = BeautifulSoup(driver.page_source, "html.parser")
            driver.quit()
            results = soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd", limit=5)
            summaries = [result.text for result in results if result.text]
            self.store_memory(summaries)
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
            self.store_memory(paragraphs)
            return paragraphs if paragraphs else ["No relevant information found on the page."]
        except Exception as e:
            return [f"An error occurred during the web scraping: {str(e)}"]

    def create_agent(self, agent_name, agent_type):
        agent = BaseAgent(agent_name, self.groq_api_key, self.model, self.chromadb_client, self.workspace_dir)
        self.agents[agent_name] = agent
        return agent

    def get_agent(self, agent_name):
        return self.agents.get(agent_name)

    def choose_next_action(self, context):
        prompt = f"Given the current context: {context}, what should be the next action to take?"
        action = self.run_conversation(prompt)
        return action

def main_loop(core_agent, project_name, num_iterations=10):
    research_agent = core_agent.create_agent("ResearchAgent", "research")
    gameplan_agent = core_agent.create_agent("GamePlanAgent", "gameplan")
    python_expert_agent = core_agent.create_agent("PythonExpertAgent", "python_expert")
    debugging_agent = core_agent.create_agent("DebuggingAgent", "debugging")
    review_agent = core_agent.create_agent("ReviewAgent", "review")

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}")
        research_results = []
        research_prompts = ["Research the best practices for creating an RPG game in Python.", "Research the best libraries for game development in Python.", "Research how to implement character creation in RPG games."]
        for prompt in research_prompts:
            search_results = research_agent.google_search(prompt)
            for url in search_results:
                scraped_content = research_agent.click_and_scrape(url)
                research_results.extend(scraped_content)
                website_description = research_agent.capture_and_describe(url)
                research_results.append(website_description)

        game_plan = gameplan_agent.run_conversation(f"Using this research data: {' '.join(research_results)}, create a game plan for an RPG project.")
        tasks = [{"description": desc, "file": file} for desc, file in [("Write a Python function to handle character creation.", "functions.py"), ("Implement the main game loop.", "main.py"), ("Create utility functions for game mechanics.", "utils.py"), ("Develop core game classes and logic.", "core.py")]]
        for task in tasks:
            python_code = python_expert_agent.run_conversation(f"Generate Python code for the following task: {task['description']}", include_memory=True)
            write_response = python_expert_agent.run_conversation(json.dumps({"file_path": f"{project_name}/src/{task['file']}", "content": python_code}))
            debugged_code = debugging_agent.run_conversation(f"Debug the following Python code: {python_code}", include_memory=True)
            reviewed_code = review_agent.run_conversation(f"Review the following Python code for quality and best practices: {debugged_code}", include_memory=True)

        # Evaluate progress and adjust course if needed
        progress_evaluation = core_agent.run_conversation(f"Evaluate the progress of the RPG game development project after {iteration + 1} iterations.")
        next_action = core_agent.choose_next_action(progress_evaluation)
        print(f"Next action: {next_action}")

    print("RPG game development project completed.")

groq_api_key, model, chromadb_client, workspace_dir = os.getenv('GROQ_API_KEY'), 'llama3-70b-8192', chromadb.Client(), "workspace"
core_agent = BaseAgent("CoreAgent", groq_api_key, model, chromadb_client, workspace_dir)

project_name = "RPG_Game"
core_agent.create_project_structure(project_name)
main_loop(core_agent, project_name)
