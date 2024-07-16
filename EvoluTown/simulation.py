import random
import numpy as np
import torch
from agent import Agent, AgentNN
from plant import Plant

class Simulation:
    def __init__(self, population_size):
        self.grid_size = 100
        self.grid = self.initialize_grid()
        self.population_size = population_size
        self.population = self.initialize_population()
        self.plants = self.initialize_plants()
        self.structures = []  # List to keep track of built structures
        self.terminal_output = []  # List to store terminal output messages
        self.time_of_day = 12  # Default value: noon
        self.weather = 5  # Default value: moderate weather
        self.tick_counter = 0  # Counter for ticks within a generation

    def initialize_grid(self):
        grid = [['empty' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < 0.1:
                    grid[i][j] = 'water'
                else:
                    grid[i][j] = 'land'
        return grid

    def get_cell(self, x, y):
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            return self.grid[x][y]
        return 'empty'

    def initialize_population(self):
        prey_count = self.population_size // 2
        predator_count = self.population_size - prey_count
        return [Agent(self, agent_type='prey') for _ in range(prey_count)] + \
               [Agent(self, agent_type='predator') for _ in range(predator_count)]

    def initialize_plants(self):
        return [Plant() for _ in range(20)]  # Adjust plant number as needed

    def run_tick(self):
        self.update_time_of_day()
        self.update_weather()
        for agent in self.population:
            agent.act()
        for plant in self.plants:
            plant.grow()
        self.population = [agent for agent in self.population if agent.alive]  # Remove dead agents
        self.tick_counter += 1

    def evolve_population(self):
        self.population = [agent for agent in self.population if agent.alive]  # Ensure only living agents remain
        if len(self.population) == 0:
            self.population = self.initialize_population()
        else:
            self.population.sort(key=lambda agent: agent.calculate_fitness(), reverse=True)
            survivors = self.population[:self.population_size // 2]
            offspring = []
            while len(offspring) < self.population_size:
                parent1, parent2 = random.sample(survivors, 2)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                offspring.append(child)
            self.population = survivors + offspring

    def crossover(self, parent1, parent2):
        child = Agent(self)
        child.brain = AgentNN()
        for child_param, param1, param2 in zip(child.brain.parameters(), parent1.brain.parameters(), parent2.brain.parameters()):
            child_param.data.copy_((param1.data + param2.data) / 2)
        return child

    def mutate(self, agent):
        mutation_rate = 0.05  # Increased mutation rate for more diversity
        with torch.no_grad():
            for param in agent.brain.parameters():
                if random.random() < mutation_rate:
                    param.add_(torch.randn(param.size()) * 0.1)

    def build_structure(self, position):
        self.structures.append(position)
        print(f"Structure built at {position}")
        self.terminal_output.append(f"Structure built at {position}")

    def find_best_location_for_town(self):
        max_score = float('-inf')
        best_location = None
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                score = self.evaluate_survivability(x, y)
                if score > max_score:
                    max_score = score
                    best_location = (x, y)
        return best_location

    def evaluate_survivability(self, x, y):
        score = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid[nx][ny] == 'land':
                        score += 1
                    elif self.grid[nx][ny] == 'water':
                        score -= 1
        return score

    def update_time_of_day(self):
        self.time_of_day = (self.time_of_day + 1) % 24

    def update_weather(self):
        self.weather = random.randint(0, 10)

    def log_generation(self, generation):
        print(f"Generation {generation} complete. Population size: {len(self.population)}")
        self.terminal_output.append(f"Generation {generation} complete. Population size: {len(self.population)}")
