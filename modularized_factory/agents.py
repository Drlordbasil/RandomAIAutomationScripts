import json
from config import SCRIPT_PATH

class Agent:
    def __init__(self, name, task):
        self.name = name
        self.task = task
        self.chat_history = []

    def get_user_prompt(self, data):
        raise NotImplementedError("Each agent must implement the get_user_prompt method.")

class UtilityAgent(Agent):
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
            self.get_user_prompt({"hr_action": "hire", "agent_name": "New Agent"})
        elif len(factory.agents) > 14:
            self.get_user_prompt({"hr_action": "fire", "agent_name": factory.agents[-1].name})

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

class ManagerAgent(Agent):
    def get_user_prompt(self, data):
        return json.dumps({
            "data": data
        })
