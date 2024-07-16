import os
import random
import math
from typing import Dict, List, Tuple

import chromadb
import ollama
from groq import Groq
from langchain_community.llms import Ollama


class OllamaModel:
    def __init__(self):
        self.model = Ollama(model="phi3:instruct")
        ollama.pull("mxbai-embed-large")

    def invoke(self, prompt: str) -> str:
        return self.model.invoke(prompt).strip()


class GroqModel:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def invoke(self, prompt: str) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
        )
        return chat_completion.choices[0].message.content.strip()


class MemoryManager:
    def __init__(self, name: str):
        self.memory_db = chromadb.Client().create_collection(name=f"{name}_memories")
        self.embed_model = "mxbai-embed-large"

    def remember(self, event: str):
        response = ollama.embeddings(model=self.embed_model, prompt=event)
        memory_id = str(self.memory_db.count())
        self.memory_db.add(
            ids=[memory_id],
            embeddings=[response["embedding"]],
            documents=[event],
        )

    def recall(self, prompt: str) -> str:
        response = ollama.embeddings(model=self.embed_model, prompt=prompt)
        results = self.memory_db.query(
            query_embeddings=[response["embedding"]],
            n_results=1,
        )
        return results["documents"][0][0] if results["documents"] and results["documents"][0] else "No relevant memory found."


class Town:
    def __init__(self):
        self.weather = "Sunny"
        self.time = "Morning"
        self.locations = {
            "Alice": {"coords": (2, 3), "role": "teacher", "skills": ["teaching", "math"], "goals": ["inspire students", "organize a workshop"]},
            "Bob": {"coords": (5, 6), "role": "farmer", "skills": ["farming", "carpentry"], "goals": ["expand farm", "build a new barn"]},
            "Charlie": {"coords": (1, 2), "role": "artist", "skills": ["painting", "sculpture"], "goals": ["create a masterpiece", "open an art gallery"]},
        }
        self.shops = {
            "bakery": {"items": ["bread", "cake", "cookie"], "open": True, "coords": (4, 5)},
            "market": {"items": ["apple", "banana", "carrot"], "open": True, "coords": (7, 8)},
            "pharmacy": {"items": ["medicine", "bandage", "vitamins"], "open": False, "coords": (3, 3)},
            "bookstore": {"items": ["novel", "magazine", "textbook"], "open": True, "coords": (6, 5)},
        }
        self.events = {
            "concert": {"coords": (2, 2), "time": "19:00"},
            "market_day": {"coords": (7, 8), "time": "10:00"},
            "workshop": {"coords": (5, 5), "time": "14:00", "skill": "carpentry"},
            "art_exhibition": {"coords": (3, 7), "time": "16:00"},
        }
        self.services = {
            "post_office": {"open": True, "services": ["mail", "package", "money order"], "coords": (8, 9)},
            "bank": {"open": True, "services": ["deposit", "withdrawal", "loan"], "coords": (1, 1)},
            "library": {"open": True, "services": ["borrow books", "study", "attend classes"], "coords": (6, 3)},
            "gym": {"open": True, "services": ["workout", "yoga", "swimming"], "coords": (4, 8)},
        }
        self.social_network = {
            "Alice": ["Bob", "Charlie"],
            "Bob": ["Alice"],
            "Charlie": ["Alice"],
        }

    def get_weather(self) -> str:
        return f"Current weather: {self.weather}"

    def change_weather(self, weather: str) -> str:
        self.weather = weather
        return f"Weather changed to {weather}"

    def get_time(self) -> str:
        return f"Current time: {self.time}"

    def change_time(self, time: str) -> str:
        self.time = time
        return f"Time changed to {time}"

    def distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> float:
        return math.hypot(loc2[0] - loc1[0], loc2[1] - loc1[1])


class Agent:
    def __init__(self, name: str, town: Town):
        self.name = name
        self.town = town
        self.coords = town.locations[name]["coords"]
        self.role = town.locations[name]["role"]
        self.skills = town.locations[name]["skills"]
        self.goals = town.locations[name]["goals"]
        self.social_connections = town.social_network[name]
        self.mood = 5
        self.reputation = 0
        self.memory = MemoryManager(name)
        self.thought_model = OllamaModel()
        self.action_model = GroqModel(os.environ.get("GROQ_API_KEY", ""))
        self.last_actions: List[str] = []
        self.initialize_memories()

    def initialize_memories(self):
        initial_memories = [
            f"{self.name} likes to visit the bakery.",
            f"{self.name} enjoys interacting with friends.",
            f"{self.name} often attends the concert in the town.",
            f"{self.name} sometimes likes to shop at the market.",
            f"{self.name} occasionally uses the post office services.",
        ]
        for event in initial_memories:
            self.memory.remember(event)

    def reflect(self) -> str:
        recent_actions = ", ".join(self.last_actions[-3:])
        current_location = f"Current location: {self.coords}"
        nearby_people = f"Nearby people: {', '.join(self.find_nearby('people'))}"
        nearby_places = f"Nearby places: {', '.join(self.find_nearby('places'))}"

        reflection_prompt = f"""
{self.name} is reflecting on their recent actions: {recent_actions}
{current_location}
{nearby_people}
{nearby_places}
Current mood: {self.mood}
Current reputation: {self.reputation}
Skills: {', '.join(self.skills)}
Goals: {', '.join(self.goals)}

Based on these actions, current situation, skills, and goals, what insights or plans should {self.name} consider for the future?
How can {self.name} improve their mood, reputation, and progress towards their goals?
Provide a thoughtful reflection considering {self.name}'s role, social connections, and the available actions they can take.
"""

        relevant_memories = self.memory.recall(reflection_prompt)
        reflection_with_memories = f"{reflection_prompt}\nRelevant memories: {relevant_memories}"

        reflection = self.thought_model.invoke(reflection_with_memories)
        self.memory.remember(f"{self.name} reflected: {reflection}")
        return reflection

    def decide_action(self) -> str:
        reasoning_prompt = f"""
{self.name} is considering their next action.
Current location: {self.coords}
Nearby people: {', '.join(self.find_nearby('people'))}
Nearby places: {', '.join(self.find_nearby('places'))}
Current time: {self.town.get_time()}
Mood: {self.mood}
Reputation: {self.reputation}
Skills: {', '.join(self.skills)}
Goals: {', '.join(self.goals)}
Social connections: {', '.join(self.social_connections)}

What thoughts or considerations should {self.name} take into account before deciding on an action?
How can {self.name} leverage their skills, social connections, and available opportunities to improve their situation and achieve their goals?
"""
        reasoning = self.thought_model.invoke(reasoning_prompt)
        self.memory.remember(f"{self.name}'s thoughts: {reasoning}")

        action_prompt = f"""
{self.name} is currently at coordinates {self.coords}.
Nearby landmarks include {', '.join(self.find_nearby('places'))}.
People nearby: {', '.join(self.find_nearby('people'))}.
Current time: {self.town.get_time()}
Previous thoughts: {reasoning}

Available commands:
- move to <x> <y>: Move to the specified coordinates.
- interact with <person>: Initiate an interaction with the specified person.
- check <shop>: Visit the specified shop and examine its items.
- attend <event>: Attend the specified event at its location.
- use <service>: Use the specified service for a specific purpose.
- learn <skill>: Attend a class or workshop to learn or improve a skill.
- work on <goal>: Take action towards achieving a specific goal.
- reflect: Reflect on recent actions, current situation, and generate insights or plans.
Visulize the town as a 2d map using ASCII art in your minds eye to visualize the town and the people in it to have spacial awareness.   
Considering {self.name}'s current situation, previous thoughts, skills, goals, and the available actions, what would be the most beneficial action to take next?
Provide reasoning for your decision.
"""

        action_decision = self.action_model.invoke(action_prompt)
        action_lines = action_decision.split("\n")
        action = action_lines[0].strip()
        reasoning = " ".join(action_lines[1:])

        self.memory.remember(f"{self.name} decided to {action} because {reasoning}")
        return action

    def find_nearby(self, type: str) -> List[str]:
        nearby = []
        if type == "people":
            for person, info in self.town.locations.items():
                if person != self.name and self.town.distance(self.coords, info["coords"]) <= 5:
                    nearby.append(person)
        else:
            for key, info in {**self.town.shops, **self.town.events, **self.town.services}.items():
                if self.town.distance(self.coords, info["coords"]) <= 5:
                    nearby.append(f"{key}")
        return nearby

    def update_mood(self, change: int):
        self.mood = max(1, min(10, self.mood + change))

    def update_reputation(self, change: int):
        self.reputation += change

    def learn_skill(self, skill: str):
        if skill not in self.skills:
            self.skills.append(skill)
            self.update_mood(1)
            self.update_reputation(1)
            return f"{self.name} learned a new skill: {skill}"
        else:
            return f"{self.name} already knows the skill: {skill}"

    def work_on_goal(self, goal: str):
        if goal in self.goals:
            self.update_mood(2)
            self.update_reputation(2)
            return f"{self.name} made progress towards the goal: {goal}"
        else:
            return f"{self.name} does not have the goal: {goal}"

    def perform_action(self) -> str:
        action = self.decide_action()
        self.memory.remember(f"{self.name} decided to {action} at {self.coords}")

        action_type, *details = action.split()
        if action_type == "move":
            self.coords = (int(details[1]), int(details[2]))
            result = f"{self.name} moved to {self.coords}."
        elif action_type == "interact":
            target = details[1]
            if target in self.town.locations:
                interaction_prompt = f"""
{self.name} is interacting with {target}, who is a {self.town.locations[target]['role']}. 
They are currently at {self.coords}. 
Visulize the town as a 2d map using ASCII art in your minds eye to visualize the town and the people in it to have spacial awareness.   

What would be an interesting and meaningful interaction between {self.name} and {target}? 
Consider their roles, skills, goals, social connection, and any relevant memories. Provide a short dialogue exchange.
"""
                interaction = self.action_model.invoke(interaction_prompt)
                result = f"{self.name} interacted with {target}:\n{interaction}"
                if target in self.social_connections:
                    self.update_mood(1)
                    self.update_reputation(1)
            else:
                result = f"{self.name} tried to interact with {target}, but they couldn't be found."
        elif action_type == "check":
            shop = details[1]
            if shop in self.town.shops:
                check_prompt = f"""
Visulize the town as a 2d map using ASCII art in your minds eye to visualize the town and the people in it to have spacial awareness.   

{self.name} is checking the {shop}, which has the following items: {', '.join(self.town.shops[shop]['items'])}.
What item might {self.name} be interested in and why? Consider {self.name}'s role, skills, goals, and any relevant memories.
"""
                check_result = self.action_model.invoke(check_prompt)
                result = f"{self.name} checked the {shop}. {check_result}"
            else:
                result = f"{self.name} tried to check the {shop}, but it doesn't exist."
        elif action_type == "attend":
            event = details[1]
            if event in self.town.events:
                attend_prompt = f"""
Visulize the town as a 2d map using ASCII art in your minds eye to visualize the town and the people in it to have spacial awareness.   

{self.name} is attending the {event} at {self.town.events[event]['coords']}. 
What might {self.name} experience or learn at this event? Consider their role, skills, goals, and interests.
"""
                attend_result = self.action_model.invoke(attend_prompt)
                result = f"{self.name} attended the {event}. {attend_result}"
                if "skill" in self.town.events[event]:
                    skill = self.town.events[event]["skill"]
                    self.learn_skill(skill)
            else:
                result = f"{self.name} tried to attend {event}, but it isn't happening."
        elif action_type == "use":
            service = details[1]
            if service in self.town.services:
                use_prompt = f"""
Visulize the town as a 2d map using ASCII art in your minds eye to visualize the town and the people in it to have spacial awareness.   

{self.name} is using the {service} to access {', '.join(self.town.services[service]['services'])}.
What specific service might {self.name} use and for what purpose? Consider their role, skills, goals, and needs.
"""
                use_result = self.action_model.invoke(use_prompt)
                result = f"{self.name} used the {service}. {use_result}"
            else:
                result = f"{self.name} tried to use {service}, but it isn't available."
        elif action_type == "learn":
            skill = details[1]
            result = self.learn_skill(skill)
        elif action_type == "work":
            goal = " ".join(details[2:])
            result = self.work_on_goal(goal)
        else:
            result = f"{self.name} is confused and does nothing."

        result = self.reflect()

        self.last_actions.append(result)
        return result


def simulate_town(num_turns: int = 10):
    town = Town()
    agents = {name: Agent(name, town) for name in town.locations}

    for turn in range(num_turns):
        print(f"\nTurn {turn + 1} - {town.get_time()}")
        for name, agent in agents.items():
            result = agent.perform_action()
            print(result)
        
        if town.time == "Morning":
            town.change_time("Afternoon")
        elif town.time == "Afternoon":
            town.change_time("Evening")
        else:
            town.change_time("Morning")


if __name__ == "__main__":
    num_turns = 10
    simulate_town(num_turns)
