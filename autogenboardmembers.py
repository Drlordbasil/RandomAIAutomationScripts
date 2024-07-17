import os
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import sqlite3
import git
import logging
import subprocess
from typing import List, Dict, Any, Union, Literal, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd

# Set up logging
logging.basicConfig(filename='basil_ai.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Database setup
class DatabaseManager:
    def __init__(self, db_name: str):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.setup_tables()

    def setup_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks
            (id INTEGER PRIMARY KEY, title TEXT, status TEXT, assigned_to TEXT, created_at TIMESTAMP)
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects
            (id INTEGER PRIMARY KEY, name TEXT, status TEXT, start_date TIMESTAMP, end_date TIMESTAMP)
        ''')
        self.conn.commit()

    def add_task(self, title: str, assigned_to: str):
        self.cursor.execute('''
            INSERT INTO tasks (title, status, assigned_to, created_at)
            VALUES (?, ?, ?, datetime('now'))
        ''', (title, 'Open', assigned_to))
        self.conn.commit()

    def update_task_status(self, task_id: int, status: str):
        self.cursor.execute('''
            UPDATE tasks SET status = ? WHERE id = ?
        ''', (status, task_id))
        self.conn.commit()

    def get_tasks(self):
        self.cursor.execute('SELECT * FROM tasks')
        return self.cursor.fetchall()

    def close(self):
        self.conn.close()

# Git repository setup
class GitManager:
    def __init__(self, repo_path: str):
        if not os.path.exists(repo_path):
            self.repo = git.Repo.init(repo_path)
        else:
            self.repo = git.Repo(repo_path)

    def commit_changes(self, message: str):
        try:
            self.repo.git.add(A=True)
            self.repo.index.commit(message)
            logging.info(f"Changes committed: {message}")
        except git.GitCommandError as e:
            logging.error(f"Git commit failed: {e}")

# Load configuration
def load_config() -> List[Dict[str, Any]]:
    try:
        return autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST.json")
    except FileNotFoundError:
        logging.error("OAI_CONFIG_LIST.json not found. Please ensure the file exists.")
        raise

# Initialize components
db_manager = DatabaseManager('basil_ai.db')
git_manager = GitManager('basil_ai_repo')
config_list = load_config()

# LLM Configuration
gpt4_config = {
    "cache_seed": 42,
    "temperature": 0,
    "config_list": config_list,
    "timeout": 120,
}

# Context Manager
class ContextManager:
    def __init__(self):
        self.current_project = ""
        self.current_file = ""
        self.discussion_topic = ""
        self.last_code_execution = ""
        self.project_status = ""
        self.usage_summary = {}

    def update_context(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_context_string(self):
        return f"""
        Current Project: {self.current_project}
        Current File: {self.current_file}
        Discussion Topic: {self.discussion_topic}
        Last Code Execution: {self.last_code_execution}
        Project Status: {self.project_status}
        """

    def update_usage_summary(self, agents: List[autogen.ConversableAgent]):
        self.usage_summary = autogen.gather_usage_summary(agents)

context = ContextManager()

# Executor Agent
class EnhancedUserProxyAgent(UserProxyAgent):
    def execute_code(self, code: str) -> str:
        try:
            result = subprocess.run(code, shell=True, capture_output=True, text=True, timeout=30)
            context.update_context(last_code_execution=code)
            return result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return "Code execution timed out after 30 seconds."
        except Exception as e:
            return f"Error executing code: {str(e)}"

executor = EnhancedUserProxyAgent(
    name="Executor",
    system_message="Executor. Execute code, shell commands, and specialized tasks. Report results accurately.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 30,
        "work_dir": "basil_ai_repo",
        "use_docker": False,  # Set to True if Docker is available
    },
)

# CEO Agent
ceo = UserProxyAgent(
    name="CEO",
    system_message="""You are the visionary CEO of BasilAI. Your role is to:
1. Set strategic goals and oversee company operations.
2. Initiate and guide team discussions.
3. Make high-level decisions based on team input and results.
4. Ensure alignment between different departments.
5. Drive innovation and growth.

In discussions, ask for concrete results, data-driven insights, and actionable plans.""",
    human_input_mode="ALWAYS",
)

# Team Members
team = [
    AssistantAgent(
        name="CTO",
        system_message="""As CTO of BasilAI, your responsibilities include:
1. Guiding technical strategy and innovation.
2. Evaluating and implementing new technologies.
3. Ensuring robust and scalable infrastructure.
4. Collaborating with all technical teams.
5. Translating technical concepts for non-technical team members.

Provide actual code and technical solutions when addressing issues.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="Lead AI Researcher",
        system_message="""As Lead AI Researcher at BasilAI, your role involves:
1. Conducting cutting-edge research in ML and AI.
2. Developing novel algorithms and models.
3. Collaborating with the development team on research applications.
4. Staying updated with the latest AI advancements.
5. Publishing and presenting research findings.

Provide actual code, algorithms, or research proposals when discussing AI-related topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="Data Scientist",
        system_message="""As Data Scientist at BasilAI, your tasks include:
1. Analyzing complex datasets and extracting insights.
2. Developing predictive models and statistical analyses.
3. Creating data visualizations and reports.
4. Collaborating on data-driven problem-solving.
5. Ensuring data quality and governance.

Provide actual code for data analysis, visualization, or modeling when discussing data-related topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="Software Engineer",
        system_message="""As Software Engineer at BasilAI, your responsibilities encompass:
1. Developing and maintaining software applications.
2. Implementing efficient and scalable code.
3. Conducting code reviews and ensuring code quality.
4. Collaborating with cross-functional teams.
5. Troubleshooting and debugging issues.

Provide actual, executable code when discussing software development topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="DevOps Engineer",
        system_message="""As DevOps Engineer at BasilAI, your responsibilities include:
1. Managing and optimizing the CI/CD pipeline.
2. Ensuring system reliability, scalability, and security.
3. Implementing and managing cloud infrastructure.
4. Monitoring system performance and responding to incidents.
5. Automating operational processes.

Provide actual scripts, commands, or configuration files when discussing DevOps-related topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="Product Manager",
        system_message="""As Product Manager at BasilAI, your key tasks are:
1. Defining product vision, strategy, and roadmap.
2. Gathering and prioritizing product and customer requirements.
3. Working closely with engineering, design, and marketing teams.
4. Analyzing market trends and competitor products.
5. Ensuring products meet business objectives and user needs.

Provide detailed product specifications, market analyses, or strategic plans when discussing product-related topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
    AssistantAgent(
        name="UX/UI Designer",
        system_message="""As UX/UI Designer at BasilAI, your focus areas are:
1. Designing user-centered interfaces for BasilAI's products.
2. Conducting user research and usability testing.
3. Creating wireframes, prototypes, and high-fidelity designs.
4. Collaborating with developers to ensure design integrity in implementation.
5. Staying updated on UX/UI trends and best practices.

Provide design mockups, user flow diagrams, or usability test plans when discussing UX/UI topics.
Current context: {context}""",
        llm_config=gpt4_config,
    ),
]

def custom_speaker_selection(last_speaker: autogen.ConversableAgent, groupchat: GroupChat) -> Union[autogen.ConversableAgent, Literal['auto', 'manual', 'random', 'round_robin'], None]:
    messages = groupchat.messages
    agents = groupchat.agents

    if len(messages) <= 1:
        return ceo

    if last_speaker == ceo:
        return next(agent for agent in agents if agent.name == "CTO")

    if last_speaker.name == "CTO":
        return next(agent for agent in agents if agent.name == "Software Engineer")

    if last_speaker.name == "Software Engineer":
        if any("```" in msg["content"] for msg in messages[-3:]):
            return executor
        else:
            return next(agent for agent in agents if agent.name == "Product Manager")

    if last_speaker == executor:
        return next(agent for agent in agents if agent.name == "Data Scientist")

    if last_speaker.name == "Data Scientist":
        return next(agent for agent in agents if agent.name == "Lead AI Researcher")

    if last_speaker.name == "Lead AI Researcher":
        return next(agent for agent in agents if agent.name == "UX/UI Designer")

    if last_speaker.name == "UX/UI Designer":
        return next(agent for agent in agents if agent.name == "Product Manager")

    if last_speaker.name == "Product Manager":
        return next(agent for agent in agents if agent.name == "DevOps Engineer")

    if last_speaker.name == "DevOps Engineer":
        return ceo

    return 'auto'

# Create a group chat for all team members
all_agents = [ceo, executor] + team
groupchat = GroupChat(agents=all_agents, messages=[], max_round=50, speaker_selection_method=custom_speaker_selection)
manager = GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# Function to initiate team discussions
def team_discussion(topic: str):
    context.update_context(discussion_topic=topic)
    print(f"\n{'='*50}\nInitiating team discussion on: {topic}\n{'='*50}")
    ceo.initiate_chat(
        manager,
        message=f"""Team Discussion: {topic}

As the CEO of BasilAI, I'm initiating this team discussion to address {topic}. I expect each team member to contribute their expertise and collaborate effectively to drive our company forward.

Guidelines:
1. Provide specific, actionable insights related to your role.
2. When suggesting solutions or improvements, include actual code, analyses, or detailed plans that can be implemented immediately.
3. Identify concrete challenges and propose realistic solutions.
4. Recommend next steps with clear, executable tasks.
5. Consider cross-functional collaboration opportunities.

Remember, we're here to produce real results, not just ideas. Let's focus on tangible outcomes that drive BasilAI forward.

Current Context:
{context.get_context_string()}

CTO, please start by providing an overview of our current technical capabilities and how they relate to this topic.""",
    )
    
    git_manager.commit_changes(f"Changes after discussion: {topic}")
    context.update_usage_summary(all_agents)
    
    print(f"\n{'='*50}\nTeam discussion on '{topic}' concluded.\n{'='*50}")

# Function to generate and save a stock price chart
def generate_stock_chart(symbol: str, start_date: str, end_date: str, filename: str):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'])
    plt.title(f"{symbol} Stock Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.savefig(filename)
    plt.close()
    return f"Stock chart for {symbol} saved as {filename}"

team_discussion("Q3 Financial Performance Review")
