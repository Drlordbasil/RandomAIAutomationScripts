import json
import threading
import time
from factory import Factory, debug_print
from agents import UtilityAgent, ResearchAgent, HRAgent, DebuggingAgent, TestingAgent, ManagerAgent

def run_factory():
    factory = Factory()

    factory.hire_agent(UtilityAgent("Utility Agent", "utility tasks"))
    factory.hire_agent(ResearchAgent("AI Research Agent", "AI research tasks", "AI advancements in 2024"))
    factory.hire_agent(ResearchAgent("NLP Research Agent", "NLP research tasks", "NLP techniques in 2024"))
    factory.hire_agent(ResearchAgent("CV Research Agent", "Computer Vision research tasks", "Computer Vision improvements in 2024"))
    factory.hire_agent(HRAgent("HR Agent", "HR tasks"))
    factory.hire_agent(DebuggingAgent("Debugging Agent", "debugging tasks"))
    factory.hire_agent(TestingAgent("Testing Agent", "testing tasks"))
    factory.hire_agent(ManagerAgent("Manager Agent", "management tasks"))

    initial_data = {
        "script": "Initial script content",
        "hr_action": "hire",
        "agent_name": "Test Agent"
    }
    while True:
        final_data = factory.start_production(initial_data)
        debug_print("Final Data: " + json.dumps(final_data, indent=2))
        time.sleep(10)

def main():
    factory_thread = threading.Thread(target=run_factory)
    factory_thread.start()
    factory_thread.join()

if __name__ == "__main__":
    main()
