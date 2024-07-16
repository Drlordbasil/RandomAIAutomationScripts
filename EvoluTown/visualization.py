import pygame
import matplotlib.pyplot as plt
from collections import deque

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
RED = (255, 0, 0)
BROWN = (139, 69, 19)

class Visualization:
    """
    Class to handle visualization of the simulation using Pygame.
    """

    def __init__(self, screen, simulation, window_width, window_height, info_panel_width, grid_size, cell_size):
        """
        Initializes the visualization class with the given parameters.

        Args:
            screen (pygame.Surface): The Pygame screen to draw on.
            simulation (Simulation): The simulation instance.
            window_width (int): The width of the window.
            window_height (int): The height of the window.
            info_panel_width (int): The width of the info panel.
            grid_size (int): The size of the grid.
            cell_size (int): The size of each cell in the grid.
        """
        self.screen = screen
        self.simulation = simulation
        self.window_width = window_width
        self.window_height = window_height
        self.info_panel_width = info_panel_width
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.font = pygame.font.SysFont('Arial', 20)
        self.output_font = pygame.font.SysFont('Arial', 14)
        self.fitness_history = deque(maxlen=100)

    def draw_info_panel(self, generation):
        """
        Draws the information panel displaying the current generation and agent details.

        Args:
            generation (int): The current generation number.
        """
        try:
            pygame.draw.rect(self.screen, WHITE, (self.window_width - self.info_panel_width, 0, self.info_panel_width, self.window_height))
            gen_text = self.font.render(f"Generation: {generation}", True, BLACK)
            self.screen.blit(gen_text, (self.window_width - self.info_panel_width + 10, 10))

            y_offset = 50
            for agent in self.simulation.population:
                if agent.alive:  # Only display living agents
                    color = self.get_agent_color(agent.genome)
                    pygame.draw.rect(self.screen, color, (self.window_width - self.info_panel_width + 10, y_offset, 20, 20))
                    agent_text = self.font.render(f"{agent.agent_type.capitalize()} - H:{agent.hunger}, T:{agent.thirst}, S:{agent.social_need}, E:{agent.energy}", True, BLACK)
                    self.screen.blit(agent_text, (self.window_width - self.info_panel_width + 40, y_offset))
                    y_offset += 30

            avg_fitness = sum(agent.calculate_fitness() for agent in self.simulation.population) / len(self.simulation.population)
            self.fitness_history.append(avg_fitness)
            self.plot_fitness()
        except Exception as e:
            print(f"An error occurred while drawing the info panel: {e}")

    def get_agent_color(self, genome):
        """
        Generates a color based on the agent's genome.

        Args:
            genome (str): The genome of the agent.

        Returns:
            tuple: A tuple representing an RGB color.
        """
        try:
            hash_value = hash(genome) % 0xFFFFFF
            return ((hash_value >> 16) & 0xFF, (hash_value >> 8) & 0xFF, hash_value & 0xFF)
        except Exception as e:
            print(f"An error occurred while generating agent color: {e}")
            return BLACK

    def draw_simulation(self):
        """
        Draws the simulation grid, plants, agents, and structures.
        """
        try:
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if self.simulation.grid[i][j] == 'water':
                        color = BLUE
                    else:
                        color = GREEN
                    pygame.draw.rect(self.screen, color, (i * self.cell_size, j * self.cell_size, self.cell_size, self.cell_size))

            for plant in self.simulation.plants:
                pygame.draw.rect(self.screen, DARK_GREEN, (plant.position[0] * self.cell_size, plant.position[1] * self.cell_size, self.cell_size, self.cell_size))

            for agent in self.simulation.population:
                if agent.alive:  # Only draw living agents
                    color = self.get_agent_color(agent.genome)
                    pygame.draw.rect(self.screen, color, (agent.position[0] * self.cell_size, agent.position[1] * self.cell_size, self.cell_size, self.cell_size))

            for structure in self.simulation.structures:
                pygame.draw.rect(self.screen, GRAY, (structure[0] * self.cell_size, structure[1] * self.cell_size, self.cell_size, self.cell_size))
        except Exception as e:
            print(f"An error occurred while drawing the simulation: {e}")

    def highlight_cell(self, position, color=RED):
        """
        Highlights a specific cell in the grid.

        Args:
            position (tuple): The position of the cell to highlight.
            color (tuple, optional): The color to highlight the cell with. Defaults to RED.
        """
        try:
            pygame.draw.rect(self.screen, color, (position[0] * self.cell_size, position[1] * self.cell_size, self.cell_size, self.cell_size))
        except Exception as e:
            print(f"An error occurred while highlighting a cell: {e}")

    def draw_terminal_output(self):
        """
        Draws the terminal output on the Pygame screen.
        """
        try:
            y_offset = self.window_height - 250  # Start drawing the terminal output from the bottom
            for line in self.simulation.terminal_output[-30:]:  # Display the last 30 lines of output
                output_text = self.output_font.render(line, True, BLACK)
                self.screen.blit(output_text, (self.window_width - self.info_panel_width + 10, y_offset))
                y_offset += 20
        except Exception as e:
            print(f"An error occurred while drawing the terminal output: {e}")

    def plot_fitness(self):
        """
        Plots the fitness history of the agents.
        """
        plt.clf()
        plt.plot(self.fitness_history)
        plt.xlabel('Generation')
        plt.ylabel('Average Fitness')
        plt.title('Fitness over Generations')
        plt.savefig('fitness_plot.png')
        fitness_plot = pygame.image.load('fitness_plot.png')
        fitness_plot = pygame.transform.scale(fitness_plot, (self.info_panel_width - 20, 200))
        self.screen.blit(fitness_plot, (self.window_width - self.info_panel_width + 10, self.window_height - 230))
