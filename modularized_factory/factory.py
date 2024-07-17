from queue import Queue
import json
import time
from rich.console import Console
from config import client, MODEL
from agents import UtilityAgent, ResearchAgent, HRAgent, DebuggingAgent, TestingAgent, ManagerAgent
from web_research import perform_web_research
from test_runner import run_tests
from hr_manager import manage_hr

console = Console()

def debug_print(message):
    console.print(f"[bold yellow][DEBUG][/bold yellow] {message}")

def run_conversation(user_prompt):
    debug_print("Starting run_conversation")
    available_functions = {
        "perform_web_research": perform_web_research,
        "run_tests": run_tests,
        "manage_hr": manage_hr
    }

    messages = [
        {
            "role": "system",
            "content": "You are a function calling LLM that uses various tools to perform tasks in a software factory. "
                       "You can call tools such as 'perform_web_research', 'run_tests', and 'manage_hr' to get your tasks done."
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

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
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
                    "content": function_response,
                }
            )

        try:
            second_response = client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
            second_response_content = second_response.choices[0].message.content
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

    def hire_agent(self, agent):
        self.agents.append(agent)
        console.print(f"[bold green]Hired {agent.name} for {agent.task}[/bold green]")

    def start_production(self, data):
        try:
            for agent in self.agents:
                self.queue.put(agent)

            while not self.queue.empty():
                agent = self.queue.get()
                user_prompt = agent.get_user_prompt(data)
                debug_print(f"Agent {agent.name} performing task: {agent.task}")
                result, updated_messages = run_conversation(user_prompt)
                debug_print(f"Result: {result}")
                if isinstance(result, str) and result.startswith("Error"):
                    raise Exception(result)
                try:
                    data.update(result if isinstance(result, dict) else {"response": result})
                except json.JSONDecodeError as e:
                    debug_print(f"Failed to decode JSON from result: {result}. Error: {str(e)}")
                except Exception as e:
                    debug_print(f"Unexpected error: {str(e)}")
                time.sleep(10)
            return data
        except Exception as e:
            debug_print(f"Error in start_production: {str(e)}")
            return f"Error in start_production: {str(e)}"
