import pygame
import openai
import random
import math
import numpy as np

# Initialize Pygame
pygame.init()

# Set up the game window
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("RPG Game")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Load game assets
player_image = pygame.Surface((50, 50))
player_image.fill(WHITE)
npc_image = pygame.Surface((50, 50))
npc_image.fill(RED)
monster_image = pygame.Surface((50, 50))
monster_image.fill(BLUE)
item_image = pygame.Surface((30, 30))
item_image.fill(GREEN)
boss_image = pygame.Surface((80, 80))
boss_image.fill(YELLOW)

# Define game world
world = {
    'starting_area': {
        'description': 'A small village surrounded by forests.',
        'neighbors': ['forest', 'mountain'],
        'items': ['shrimp', 'axe', 'health_potion'],
        'monsters': ['goblin']
    },
    'forest': {
        'description': 'A dense forest filled with mysteries.',
        'neighbors': ['starting_area', 'river', 'cave'],
        'items': ['healing_potion', 'sword'],
        'monsters': ['wolf', 'bear']
    },
    'mountain': {
        'description': 'A towering mountain with treacherous paths.',
        'neighbors': ['starting_area', 'cave'],
        'items': ['armor'],
        'monsters': ['eagle', 'troll']
    },
    'river': {
        'description': 'A calm river flowing through the forest.',
        'neighbors': ['forest'],
        'items': ['fishing_rod', 'fish'],
        'monsters': ['serpent', 'alligator']
    },
    'cave': {
        'description': 'A dark and mysterious cave.',
        'neighbors': ['forest', 'mountain'],
        'items': ['treasure'],
        'monsters': ['dragon']
    }
}

class Player(pygame.sprite.Sprite):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.image = player_image
        self.rect = self.image.get_rect()
        self.rect.center = (screen_width // 2, screen_height // 2)
        self.location = 'starting_area'
        self.inventory = []
        self.health = 100
        self.energy = 100

    def move(self, direction):
        if direction == 'north':
            self.rect.y -= 5
            self.energy -= 1
        elif direction == 'south':
            self.rect.y += 5
            self.energy -= 1
        elif direction == 'west':
            self.rect.x -= 5
            self.energy -= 1
        elif direction == 'east':
            self.rect.x += 5
            self.energy -= 1
        print(f"Player moved {direction}. Energy: {self.energy}")

    def take_item(self, item_name):
        if item_name in world[self.location]['items']:
            self.inventory.append(item_name)
            world[self.location]['items'].remove(item_name)
            print(f"You took the {item_name}.")
        else:
            print(f"There is no {item_name} here to take.")

    def use_item(self, item_name):
        if item_name in self.inventory:
            if item_name == 'healing_potion':
                self.health = min(100, self.health + 20)
                self.inventory.remove(item_name)
                print("You used a healing potion and regained some health.")
            elif item_name == 'health_potion':
                self.health = min(100, self.health + 50)
                self.inventory.remove(item_name)
                print("You used a health potion and regained a lot of health.")
            elif item_name == 'shrimp':
                self.health = min(100, self.health + 4)
                self.inventory.remove(item_name)
                print("You ate some shrimp and gained 4 health.")
            else:
                print(f"You can't use the {item_name} right now.")
        else:
            print(f"You don't have a {item_name}.")

    def attack_monster(self, monster):
        if monster.name in world[self.location]['monsters']:
            damage = random.randint(10, 20)
            monster.health -= damage
            print(f"You attack the {monster.name} and deal {damage} damage.")
            if monster.health <= 0:
                print(f"You defeated the {monster.name}!")
                monsters.remove(monster)
        else:
            print(f"There is no {monster.name} here to attack.")

    def check_status(self):
        print(f"Health: {self.health}")
        print(f"Energy: {self.energy}")
        print(f"Inventory: {', '.join(self.inventory)}")

class AIPlayer(pygame.sprite.Sprite):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.image = npc_image
        self.rect = self.image.get_rect()
        self.rect.center = (screen_width // 2 + 100, screen_height // 2)
        self.location = 'starting_area'
        self.health = 100
        self.energy = 100
        self.max_energy = 100
        self.energy_regeneration = 0.1
        self.client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    def move_towards_player(self, player):
        dx = player.rect.x - self.rect.x
        dy = player.rect.y - self.rect.y
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > 50:
            speed = 2
            self.rect.x += int(dx * speed / distance)
            self.rect.y += int(dy * speed / distance)
            self.energy -= 0.1
            if self.energy < 0:
                self.energy = 0
            print(f"AI Player moved towards the player. Energy: {self.energy:.2f}")

    def regenerate_energy(self):
        self.energy = min(self.energy + self.energy_regeneration, self.max_energy)

    def attack_monster(self, monster):
        if monster.name in world[self.location]['monsters']:
            damage = random.randint(5, 10)
            monster.health -= damage
            print(f"AI Player attacks the {monster.name} and deals {damage} damage.")
            if monster.health <= 0:
                print(f"AI Player defeated the {monster.name}!")
                monsters.remove(monster)
        else:
            print(f"There is no {monster.name} here for the AI Player to attack.")

    def generate_response(self, prompt):
        response = self.client.completions.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()

class Monster(pygame.sprite.Sprite):
    def __init__(self, name, health):
        super().__init__()
        self.name = name
        self.image = monster_image
        self.rect = self.image.get_rect()
        self.health = health

class Boss(pygame.sprite.Sprite):
    def __init__(self, name, health):
        super().__init__()
        self.name = name
        self.image = boss_image
        self.rect = self.image.get_rect()
        self.health = health

class Item(pygame.sprite.Sprite):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.image = item_image
        self.rect = self.image.get_rect()

def draw_game_world():
    screen.fill(BLACK)
    # Draw the player
    screen.blit(player.image, player.rect)
    # Draw the AI player
    screen.blit(ai_player.image, ai_player.rect)
    # Draw monsters
    for monster in monsters:
        screen.blit(monster.image, monster.rect)
    # Draw items
    for item in items:
        screen.blit(item.image, item.rect)
    # Draw UI elements
    font = pygame.font.Font(None, 24)
    health_text = font.render(f"Player Health: {player.health}", True, WHITE)
    energy_text = font.render(f"Player Energy: {player.energy:.2f}", True, WHITE)
    ai_health_text = font.render(f"AI Player Health: {ai_player.health}", True, WHITE)
    ai_energy_text = font.render(f"AI Player Energy: {ai_player.energy:.2f}", True, WHITE)
    inventory_text = font.render(f"Inventory: {', '.join(player.inventory)}", True, WHITE)
    screen.blit(health_text, (10, 10))
    screen.blit(energy_text, (10, 40))
    screen.blit(ai_health_text, (10, 70))
    screen.blit(ai_energy_text, (10, 100))
    screen.blit(inventory_text, (10, 130))

    # Draw mini-map
    mini_map_size = 100
    mini_map_rect = pygame.Rect(screen_width - mini_map_size - 10, 10, mini_map_size, mini_map_size)
    pygame.draw.rect(screen, WHITE, mini_map_rect, 1)
    player_pos = (player.rect.x // 10, player.rect.y // 10)
    pygame.draw.circle(screen, RED, (mini_map_rect.x + player_pos[0], mini_map_rect.y + player_pos[1]), 2)

    # Draw keybind list
    keybind_text = font.render("Keybinds:", True, WHITE)
    screen.blit(keybind_text, (screen_width - 150, 10))
    keybind_list = [
        "Arrow Keys: Move",
        "T: Take Item",
        "U: Use Item",
        "A: Attack Monster",
        "S: Check Status"
    ]
    for i, keybind in enumerate(keybind_list):
        text = font.render(keybind, True, WHITE)
        screen.blit(text, (screen_width - 150, 40 + i * 30))

    pygame.display.flip()

def handle_input():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                player.move('north')
            elif event.key == pygame.K_DOWN:
                player.move('south')
            elif event.key == pygame.K_LEFT:
                player.move('west')
            elif event.key == pygame.K_RIGHT:
                player.move('east')
            elif event.key == pygame.K_t:
                for item in items:
                    if pygame.sprite.collide_rect(player, item):
                        player.take_item(item.name)
                        items.remove(item)
                        break
            elif event.key == pygame.K_u:
                if player.inventory:
                    player.use_item(player.inventory[0])
            elif event.key == pygame.K_a:
                for monster in monsters:
                    if pygame.sprite.collide_rect(player, monster):
                        player.attack_monster(monster)
                        break
            elif event.key == pygame.K_s:
                player.check_status()

def update_game_state():
    # Update AI player movement and attack
    ai_player.move_towards_player(player)
    ai_player.regenerate_energy()

    for monster in monsters:
        if pygame.sprite.collide_rect(ai_player, monster):
            ai_player.attack_monster(monster)
            break

    # Update item positions
    for item in items:
        item.rect.x += random.randint(-1, 1)
        item.rect.y += random.randint(-1, 1)

    # Check if player collides with boss
    for boss in bosses:
        if pygame.sprite.collide_rect(player, boss):
            print("You encountered the boss!")
            # TODO: Implement boss battle mechanics

def game_loop():
    global player, ai_player, monsters, items, bosses

    player = Player("Anthony")
    ai_player = AIPlayer("AI Companion")

    monsters = pygame.sprite.Group()
    items = pygame.sprite.Group()
    bosses = pygame.sprite.Group()

    # Create monsters
    goblin = Monster("goblin", 50)
    goblin.rect.x = 200
    goblin.rect.y = 300
    monsters.add(goblin)

    wolf = Monster("wolf", 70)
    wolf.rect.x = 400
    wolf.rect.y = 200
    monsters.add(wolf)

    # Create items
    shrimp = Item("shrimp")
    shrimp.rect.x = 400
    shrimp.rect.y = 300
    items.add(shrimp)

    health_potion = Item("health_potion")
    health_potion.rect.x = 500
    health_potion.rect.y = 400
    items.add(health_potion)

    # Create boss
    dragon = Boss("dragon", 200)
    dragon.rect.x = 600
    dragon.rect.y = 500
    bosses.add(dragon)

    clock = pygame.time.Clock()

    while True:
        handle_input()
        update_game_state()
        draw_game_world()
        clock.tick(60)  # Limit the frame rate to 60 FPS

if __name__ == "__main__":
    game_loop()
