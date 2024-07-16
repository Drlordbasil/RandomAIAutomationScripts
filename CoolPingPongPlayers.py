import numpy as np
import gym
import pygame

class AlphaGoLearning:
    def __init__(self, num_actions, learning_rate=0.91, discount_factor=0.99, exploration_rate=0.51):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = np.zeros((num_actions,))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.num_actions)
        else:
            action = np.argmax(self.q_table)
        return action

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[action]
        max_future_q = np.max(self.q_table)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * max_future_q)
        self.q_table[action] = new_q

    def train(self, num_episodes, env, opponent):
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, _ = env.step(action, opponent_action)
                self.update_q_table(state, action, reward, next_state)
                opponent.update_q_table(state, opponent_action, -reward, next_state)
                state = next_state

    def evaluate(self, env, opponent, num_episodes):
        total_rewards = 0
        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_rewards = 0
            while not done:
                action = np.argmax(self.q_table)
                opponent_action = opponent.choose_action(state)
                next_state, reward, done, _ = env.step(action, opponent_action)
                episode_rewards += reward
                state = next_state
            total_rewards += episode_rewards
        average_reward = total_rewards / num_episodes
        return average_reward
class PongEnv:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.paddle_width = 10
        self.paddle_height = 60
        self.paddle_speed = 5
        self.ball_size = 10
        self.ball_speed_x = 3
        self.ball_speed_y = 3
        self.left_paddle_y = self.height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2 - self.ball_size // 2
        self.ball_y = self.height // 2 - self.ball_size // 2
        self.left_score = 0
        self.right_score = 0

    def reset(self):
        self.left_paddle_y = self.height // 2 - self.paddle_height // 2
        self.right_paddle_y = self.height // 2 - self.paddle_height // 2
        self.ball_x = self.width // 2 - self.ball_size // 2
        self.ball_y = self.height // 2 - self.ball_size // 2
        self.ball_speed_x = 3
        self.ball_speed_y = 3
        self.left_score = 0
        self.right_score = 0
        return (self.left_paddle_y, self.right_paddle_y, self.ball_x, self.ball_y)

    def step(self, left_action, right_action):
        # Update paddles based on actions
        if left_action == 0:
            self.left_paddle_y -= self.paddle_speed
        elif left_action == 1:
            self.left_paddle_y += self.paddle_speed
        if right_action == 0:
            self.right_paddle_y -= self.paddle_speed
        elif right_action == 1:
            self.right_paddle_y += self.paddle_speed

        # Keep paddles within the screen boundaries
        self.left_paddle_y = max(0, min(self.left_paddle_y, self.height - self.paddle_height))
        self.right_paddle_y = max(0, min(self.right_paddle_y, self.height - self.paddle_height))

        # Update ball position
        self.ball_x += self.ball_speed_x
        self.ball_y += self.ball_speed_y

        # Check for collision with paddles
        if (
            self.ball_x <= self.paddle_width
            and self.left_paddle_y <= self.ball_y <= self.left_paddle_y + self.paddle_height
        ):
            self.ball_speed_x = -self.ball_speed_x
        elif (
            self.ball_x >= self.width - self.paddle_width - self.ball_size
            and self.right_paddle_y <= self.ball_y <= self.right_paddle_y + self.paddle_height
        ):
            self.ball_speed_x = -self.ball_speed_x

        # Check for collision with top/bottom walls
        if self.ball_y <= 0 or self.ball_y >= self.height - self.ball_size:
            self.ball_speed_y = -self.ball_speed_y

        # Check for scoring
        reward = 0
        done = False
        if self.ball_x <= 0:
            self.right_score += 1
            reward = -10
            done = True
        elif self.ball_x >= self.width - self.ball_size:
            self.left_score += 1
            reward = 10
            done = True

        return (self.left_paddle_y, self.right_paddle_y, self.ball_x, self.ball_y), reward, done

    def render(self, screen):
        screen.fill((0, 0, 0))
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (0, self.left_paddle_y, self.paddle_width, self.paddle_height),
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.width - self.paddle_width, self.right_paddle_y, self.paddle_width, self.paddle_height),
        )
        pygame.draw.rect(
            screen,
            (255, 255, 255),
            (self.ball_x, self.ball_y, self.ball_size, self.ball_size),
        )
        pygame.display.flip()

# Create the Pong environment
env = PongEnv()

# Create two instances of AlphaGoLearning
left_player = AlphaGoLearning(num_actions=2)
right_player = AlphaGoLearning(num_actions=2)

# Set up the Pygame window
screen = pygame.display.set_mode((env.width, env.height))
clock = pygame.time.Clock()

# Training loop
num_episodes = 10000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        left_action = left_player.choose_action(state)
        right_action = right_player.choose_action(state)
        next_state, reward, done = env.step(left_action, right_action)
        left_player.update_q_table(state, left_action, reward, next_state)
        right_player.update_q_table(state, right_action, -reward, next_state)
        state = next_state

        # Render the game
        env.render(screen)
        clock.tick(60)

        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    print(f"Episode: {episode+1}, Left Score: {env.left_score}, Right Score: {env.right_score}")

# Close the Pygame window
pygame.quit()
