import pygame
from simulation import Simulation
from visualization import Visualization

pygame.init()

# Define the window dimensions
window_width = 1200
window_height = 800
info_panel_width = 300

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create a Pygame window
screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
pygame.display.set_caption("Agent-Based Simulation")

# Initialize the simulation and visualization
simulation = Simulation(population_size=100)
visualization = Visualization(screen, simulation, window_width, window_height, info_panel_width, grid_size=100, cell_size=8)

# Main loop
running = True
generation = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            window_width, window_height = event.size
            screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
            visualization.window_width = window_width
            visualization.window_height = window_height

    screen.fill(BLACK)
    simulation.run_tick()
    visualization.draw_simulation()
    visualization.draw_info_panel(generation)
    visualization.draw_terminal_output()
    pygame.display.flip()

    if simulation.tick_counter >= 100:  # Run 100 ticks per generation
        simulation.evolve_population()
        simulation.tick_counter = 0
        generation += 1

pygame.quit()
