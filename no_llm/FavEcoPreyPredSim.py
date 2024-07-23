import pygame
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
import math

class Constants:
    EMPTY, PREY, PREDATOR, PLANT, WATER, FOOD, GOLDILOCKS, DEAD_ZONE = range(8)
    ENERGY_THRESHOLD, REPRODUCE_ENERGY, INITIAL_ENERGY = 100, 50, 25
    GRID_SIZE, NN_INPUT_SIZE, NN_OUTPUT_SIZE, TRAINING_EPOCHS = 30, 24, 6, 50
    MUTATION_CHANCE, DEATH_AFTER_REPRODUCTION_CHANCE, GENERATIONS = 0.2, 0.1, 500
    MAX_THREADS, CELL_SIZE = 10000, 30
    SIDE_PANEL_WIDTH = 200
    WINDOW_SIZE = (GRID_SIZE * CELL_SIZE + SIDE_PANEL_WIDTH, GRID_SIZE * CELL_SIZE)
    FONT_SIZE, EYE_RANGE = 12, 5
    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    WATER_LOCATIONS = [(0, 0), (0, GRID_SIZE - 1), (GRID_SIZE - 1, 0), (GRID_SIZE - 1, GRID_SIZE - 1),
                       (GRID_SIZE // 2, GRID_SIZE // 2), (GRID_SIZE // 3, GRID_SIZE // 3),
                       (2 * GRID_SIZE // 3, 2 * GRID_SIZE // 3)]
    ENTITY_COLORS = {EMPTY: (255, 255, 255), PREY: (0, 255, 0), PREDATOR: (255, 0, 0),
                     PLANT: (0, 128, 0), WATER: (0, 0, 255), FOOD: (139, 69, 19),
                     GOLDILOCKS: (255, 215, 0), DEAD_ZONE: (105, 105, 105)}
    ENTITY_LABELS = {EMPTY: 'Empty', PREY: 'Prey', PREDATOR: 'Predator', PLANT: 'Plant', WATER: 'Water',
                     FOOD: 'Food', GOLDILOCKS: 'Goldi', DEAD_ZONE: 'DeadZone'}

class NeuralNetwork(nn.Module):
    def __init__(self, parent_brain=None):
        super(NeuralNetwork, self).__init__()
        self.init_layers(parent_brain)

    def init_layers(self, parent_brain):
        if parent_brain and isinstance(parent_brain, NeuralNetwork):
            self.layers = parent_brain.layers
            self.mutate() if random.random() < Constants.MUTATION_CHANCE else None
        else:
            self.layers = nn.Sequential(
                nn.Linear(Constants.NN_INPUT_SIZE, 32), nn.ReLU(), nn.Linear(32, 16),
                nn.Tanh(), nn.Linear(16, Constants.NN_OUTPUT_SIZE), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)

    def mutate(self):
        with torch.no_grad():
            for name, param in self.layers.named_parameters():
                if random.random() < Constants.MUTATION_CHANCE:
                    param.data += torch.randn(param.size()) * 0.1 if 'bias' in name else torch.randn(param.size()) * 0.1

    def get_genetic_code(self):
        return torch.cat([param.data.flatten() for name, param in self.layers.named_parameters()])

class Trainer:
    @staticmethod
    def train(network, data):
        criterion, learning_rate = nn.CrossEntropyLoss(), 0.2
        for epoch in range(Constants.TRAINING_EPOCHS):
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)
            for state, action in data:
                optimizer.zero_grad()
                output, loss = network(state), criterion(network(state), torch.tensor([action]))
                loss.backward()
                optimizer.step()
            learning_rate *= 0.9 if epoch % 100 == 0 and epoch > 0 else 1

class Entity:
    def __init__(self, entity_type, energy=Constants.INITIAL_ENERGY, age=0, brain=None, adaptability_factor=1.0):
        self.type, self.energy, self.age = entity_type, energy, age
        self.brain = NeuralNetwork(Constants.NN_INPUT_SIZE) if brain is None else brain
        self.adaptability_factor = adaptability_factor
    
    def act(self, i, j, grid, new_grid):
        self.age += 1
        if self.age > 80:
            new_grid[i][j] = Entity(Constants.EMPTY)
            return
        if self.type == Constants.WATER:
            new_grid[i][j] = self
            return

        self.energy -= 1
        if self.energy <= 0:
            new_grid[i][j] = Entity(Constants.FOOD)
            return

        self.perform_action(self.brain(get_surrounding_state(i, j, grid)).argmax().item(), i, j, grid, new_grid)
        if self.type == Constants.PLANT or self.energy >= Constants.REPRODUCE_ENERGY:
            self.reproduce_entity(i, j, grid, new_grid)

    def perform_action(self, action, i, j, grid, new_grid):
        if action == 1:
            self.move_entity(i, j, new_grid)
        elif action == 2:
            self.eat_entity(i, j, grid, new_grid)
        elif action == 3:
            self.reproduce_entity(i, j, grid, new_grid)

    def move_entity(self, i, j, new_grid):
        dir_idx = random.choice(range(len(Constants.DIRECTIONS)))
        new_i, new_j = (i + Constants.DIRECTIONS[dir_idx][0]) % Constants.GRID_SIZE, (j + Constants.DIRECTIONS[dir_idx][1]) % Constants.GRID_SIZE
        if new_grid[new_i][new_j].type == Constants.EMPTY:
            new_grid[new_i][new_j] = Entity(self.type, self.energy, self.age, self.brain, self.adaptability_factor)
            new_grid[i][j] = Entity(Constants.EMPTY)
        elif new_grid[new_i][new_j].type == Constants.FOOD:
            self.energy += 10
            new_grid[new_i][new_j] = Entity(self.type, self.energy, self.age, self.brain, self.adaptability_factor)
            new_grid[i][j] = Entity(Constants.EMPTY)
        elif new_grid[new_i][new_j].type == Constants.PREDATOR and self.type == Constants.PREY:
            self.energy -= 10
            new_grid[new_i][new_j] = Entity(self.type, self.energy, self.age, self.brain, self.adaptability_factor)
            new_grid[i][j] = Entity(Constants.EMPTY)
        elif new_grid[new_i][new_j].type == Constants.PREY and self.type == Constants.PREDATOR:
            self.energy += 10
            new_grid[new_i][new_j] = Entity(self.type, self.energy, self.age, self.brain, self.adaptability_factor)
            new_grid[i][j] = Entity(Constants.EMPTY)
            
        else:
            new_grid[i][j] = self

    def eat_entity(self, i, j, grid, new_grid):
        for di, dj in Constants.DIRECTIONS:
            ni, nj = (i + di) % Constants.GRID_SIZE, (j + dj) % Constants.GRID_SIZE
            target_entity = grid[ni][nj]
            if self.can_eat(target_entity):
                self.energy += 15 if self.type == Constants.PREDATOR else 5
                new_grid[ni][nj] = Entity(Constants.EMPTY)
                return
        new_grid[i][j] = self

    def can_eat(self, target_entity):
        return (self.type == Constants.PREDATOR and target_entity.type in [Constants.PREY, Constants.PLANT]) or \
               (self.type == Constants.PREY and target_entity.type == Constants.PLANT)

    def reproduce_entity(self, i, j, grid, new_grid):
        reproduction_cost = Constants.REPRODUCE_ENERGY
        if self.energy >= reproduction_cost and random.random() < 0.5:
            self.energy -= reproduction_cost
            child_brain = NeuralNetwork(parent_brain=self.brain)
            child_adaptability_factor = self.adaptability_factor * (1 + random.uniform(-0.1, 0.1))
            new_grid[i][j] = Entity(self.type, Constants.INITIAL_ENERGY, 0, child_brain, child_adaptability_factor)
            if self.type != Constants.PLANT and random.random() < Constants.DEATH_AFTER_REPRODUCTION_CHANCE:
                new_grid[i][j] = Entity(Constants.EMPTY)

    def get_color_based_on_genetics(self):
        genetic_code = self.brain.get_genetic_code()
        # Normalize the genetic code values to be between 0 and 1
        normalized_code = (genetic_code - genetic_code.min()) / (genetic_code.max() - genetic_code.min())
        # Use normalized values to generate an RGB color
        r = int(normalized_code.mean().item() * 255)
        g = int((1 - normalized_code.mean().item()) * 255)
        b = int((normalized_code.std().item()) * 255)  # Using standard deviation for blue value
        return (r, g, b)

def get_surrounding_state(i, j, grid):
    state = [grid[i][j].type, grid[i][j].energy] + [grid[(i + di) % Constants.GRID_SIZE][(j + dj) % Constants.GRID_SIZE].type for di, dj in Constants.DIRECTIONS] + \
            [grid[(i + di) % Constants.GRID_SIZE][(j + dj) % Constants.GRID_SIZE].energy for di, dj in Constants.DIRECTIONS]
    state.extend([0] * (Constants.NN_INPUT_SIZE - len(state)))
    return torch.tensor(state, dtype=torch.float)

def update_grid(grid, new_grid, i_start, i_end):
    for i in range(i_start, i_end):
        for j in range(Constants.GRID_SIZE):
            entity = grid[i][j]
            if entity.type != Constants.WATER:
                entity.act(i, j, grid, new_grid)
            else:
                new_grid[i][j] = entity

def initialize_grid():
    grid = []
    for i in range(Constants.GRID_SIZE):
        row = []
        for j in range(Constants.GRID_SIZE):
            # Randomly choose an entity type with a bias towards EMPTY for initial sparsity
            entity_type = np.random.choice([Constants.EMPTY, Constants.PREY, Constants.PREDATOR, Constants.PLANT], 
                                           p=[0.1, 0.4, 0.4, 0.1])  # Adjust probabilities as needed
            if (i, j) in Constants.WATER_LOCATIONS:
                entity_type = Constants.WATER
            row.append(Entity(entity_type))
        grid.append(row)
    return grid

def draw_side_panel(screen, font):
    base_x = Constants.GRID_SIZE * Constants.CELL_SIZE
    pygame.draw.rect(screen, (230, 230, 230), (base_x, 0, Constants.SIDE_PANEL_WIDTH, Constants.WINDOW_SIZE[1]))

    entity_types = [Constants.PREY, Constants.PREDATOR, Constants.PLANT, Constants.WATER]
    y_offset = 20
    for entity_type in entity_types:
        text_surface = font.render(Constants.ENTITY_LABELS[entity_type], True, (0, 0, 0))
        screen.blit(text_surface, (base_x + 10, y_offset))
        color_rect = pygame.Rect(base_x + 150, y_offset, 20, 20)
        pygame.draw.rect(screen, Constants.ENTITY_COLORS[entity_type], color_rect)
        y_offset += 30

def run_simulation():
    pygame.init()
    screen, clock = pygame.display.set_mode(Constants.WINDOW_SIZE), pygame.time.Clock()
    pygame.display.set_caption('Fun AI (NN and RL) Simulation')
    font = pygame.font.SysFont('arial', Constants.FONT_SIZE)

    running, grid = True, initialize_grid()
    generation_time = 30
    generation_timer = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear screen
        for i in range(Constants.GRID_SIZE):
            for j in range(Constants.GRID_SIZE):
                entity = grid[i][j]
                color = Constants.ENTITY_COLORS[entity.type]
                rect = pygame.Rect(j * Constants.CELL_SIZE, i * Constants.CELL_SIZE, Constants.CELL_SIZE, Constants.CELL_SIZE)
                pygame.draw.rect(screen, color, rect)
        # Draw side panel
        draw_side_panel(screen, font)

        pygame.display.flip()
        clock.tick(60)

        generation_timer += clock.get_time()
        if generation_timer >= generation_time * 1000:
            generation_timer = 0
            grid = initialize_grid()

        new_grid = [[Entity(Constants.EMPTY) for _ in range(Constants.GRID_SIZE)] for _ in range(Constants.GRID_SIZE)]
        with ThreadPoolExecutor(max_workers=Constants.MAX_THREADS) as executor:
            futures = [executor.submit(update_grid, grid, new_grid, i * (Constants.GRID_SIZE // Constants.MAX_THREADS), (i + 1) * (Constants.GRID_SIZE // Constants.MAX_THREADS) if i != Constants.MAX_THREADS - 1 else Constants.GRID_SIZE) for i in range(Constants.MAX_THREADS)]
            for future in futures:
                future.result()
        grid = new_grid.copy()

    pygame.quit()
run_simulation()
