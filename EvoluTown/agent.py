import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from plant import Plant

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities, default=1.0)
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def sample(self, batch_size, beta=0.4):
        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.concatenate(batch[0])
        actions = batch[1]
        rewards = batch[2]
        next_states = np.concatenate(batch[3])
        dones = batch[4]

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class AgentNN(nn.Module):
    def __init__(self):
        super(AgentNN, self).__init__()
        self.fc1 = nn.Linear(14, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 9)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.fc1(x)
        if x.shape[0] > 1:
            x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        if x.shape[0] > 1:
            x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        if x.shape[0] > 1:
            x = self.bn3(x)
        x = torch.relu(x)

        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

    def from_genome(self, genome):
        weights = [int(genome[i:i+2], 16) / 255.0 for i in range(0, len(genome), 2)]
        offset = 0

        def set_weights(layer, weight_size, bias_size):
            nonlocal offset
            weight_values = weights[offset:offset + weight_size]
            bias_values = weights[offset + weight_size:offset + weight_size + bias_size]
            offset += weight_size + bias_size
            layer.weight.data = torch.tensor(weight_values).reshape(layer.weight.shape)
            layer.bias.data = torch.tensor(bias_values).reshape(layer.bias.shape)

        set_weights(self.fc1, 256 * 14, 256)
        set_weights(self.fc2, 256 * 256, 256)
        set_weights(self.fc3, 256 * 128, 128)
        set_weights(self.fc4, 128 * 64, 64)
        set_weights(self.fc5, 64 * 9, 9)

class Agent:
    def __init__(self, simulation, genome=None, agent_type='prey'):
        try:
            self.simulation = simulation
            self.position = (random.randint(0, 100), random.randint(0, 100))
            self.hunger = 100
            self.thirst = 100
            self.social_need = 100
            self.energy = 100
            self.resources = 0
            self.building_skill = 1
            self.trade_skill = 1
            self.vision_range = 5
            self.alive = True
            self.fitness = 0
            self.agent_type = agent_type
            self.genome = genome or self.generate_random_genome()
            self.brain = self.build_model()
            self.target_brain = self.build_model()
            self.optimizer = optim.Adam(self.brain.parameters(), lr=0.001)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
            self.epsilon = 1.0
            self.gamma = 0.99
            self.memory = PrioritizedReplayBuffer(10000)
            self.batch_size = 64
            self.update_target_steps = 1000
            self.step_counter = 0
        except Exception as e:
            print(f"An error occurred while initializing the agent: {e}")

    def generate_random_genome(self):
        genome_length = (256 * 14 + 256) + (256 * 256 + 256) + (256 * 128 + 128) + (128 * 64 + 64) + (64 * 9 + 9)
        return ''.join(random.choices('0123456789ABCDEF', k=genome_length * 2))

    def build_model(self):
        model = AgentNN()
        if self.genome:
            model.from_genome(self.genome)
        return model

    def act(self):
        try:
            if not self.alive:
                return
            self.move()
            if self.agent_type == 'prey':
                self.gather_resources()
                self.build_structure()
                self.trade()
                self.foraging()
                self.find_mate()
                self.social_interaction()
            elif self.agent_type == 'predator':
                self.hunt()
            self.energy -= 1
            if self.energy <= 0 or self.hunger <= 0 or self.thirst <= 0:
                self.alive = False
        except Exception as e:
            print(f"An error occurred while the agent was acting: {e}")
            self.simulation.terminal_output.append(f"An error occurred while the agent was acting: {e}")

    def move(self):
        try:
            state = self.get_state()
            action = self.select_action(state)
            next_state, reward, done = self.execute_action(action)
            self.memory.push(state, action, reward, next_state, done)
            self.learn()
            self.update_target_network()
        except Exception as e:
            print(f"An error occurred while the agent was moving: {e}")
            self.simulation.terminal_output.append(f"An error occurred while the agent was moving: {e}")

    def get_state(self):
        try:
            return np.array([
                self.hunger / 100,
                self.thirst / 100,
                self.social_need / 100,
                self.energy / 100,
                self.vision_range,
                self.resources / 100,
                self.building_skill / 10,
                self.trade_skill / 10,
                self.position[0] / self.simulation.grid_size,
                self.position[1] / self.simulation.grid_size,
                len([agent for agent in self.simulation.population if agent.agent_type == 'prey']),
                len([agent for agent in self.simulation.population if agent.agent_type == 'predator']),
                self.simulation.time_of_day / 24,
                self.simulation.weather / 10
            ], dtype=np.float32)
        except Exception as e:
            print(f"An error occurred while collecting the agent's state: {e}")
            self.simulation.terminal_output.append(f"An error occurred while collecting the agent's state: {e}")
            return np.zeros(14, dtype=np.float32)

    def select_action(self, state):
        try:
            if np.random.rand() <= self.epsilon:
                return random.choice(range(9))
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.brain(state_tensor)
            return torch.argmax(q_values).item()
        except Exception as e:
            print(f"An error occurred while selecting an action: {e}")
            self.simulation.terminal_output.append(f"An error occurred while selecting an action: {e}")
            return random.choice(range(9))

    def execute_action(self, action):
        try:
            reward = -1  # Default reward for moving
            done = False
            next_state = self.get_state()
            if action == 0:
                self.position = (self.position[0], self.position[1] + 1)
            elif action == 1:
                self.position = (self.position[0], self.position[1] - 1)
            elif action == 2:
                self.position = (self.position[0] - 1, self.position[1])
            elif action == 3:
                self.position = (self.position[0] + 1, self.position[1])
            elif action == 4:
                self.gather_resources()
                reward = 10
            elif action == 5:
                self.build_structure()
                reward = 20
            elif action == 6:  # New action: Rest
                self.rest()
                reward = 5
            elif action == 7:  # New action: Explore
                self.explore()
                reward = 5
            elif action == 8:  # New action: Mate
                self.mate()
                reward = 30
            if self.energy <= 0 or self.hunger <= 0 or self.thirst <= 0:
                done = True
            return next_state, reward, done
        except Exception as e:
            print(f"An error occurred while executing an action: {e}")
            self.simulation.terminal_output.append(f"An error occurred while executing an action: {e}")
            return self.get_state(), -1, True

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        state, action, reward, next_state, done, weights, indices = self.memory.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        q_values = self.brain(state)
        next_q_values = self.target_brain(next_state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - expected_q_value.detach()).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.memory.update_priorities(indices, prios.data.cpu().numpy())

        self.epsilon = max(0.01, self.epsilon * 0.995)  # Decrease epsilon

    def update_target_network(self):
        self.step_counter += 1
        if self.step_counter % self.update_target_steps == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())

    def get_reward(self):
        try:
            reward = 0
            if self.hunger > 0:
                reward += 1
            if self.thirst > 0:
                reward += 1
            if self.social_need > 0:
                reward += 1
            if self.energy > 0:
                reward += 1
            return reward
        except Exception as e:
            print(f"An error occurred while calculating the reward: {e}")
            self.simulation.terminal_output.append(f"An error occurred while calculating the reward: {e}")
            return 0

    def gather_resources(self):
        try:
            self.resources += 1
        except Exception as e:
            print(f"An error occurred while gathering resources: {e}")
            self.simulation.terminal_output.append(f"An error occurred while gathering resources: {e}")

    def build_structure(self):
        try:
            if self.resources >= 10:
                best_location = self.simulation.find_best_location_for_town()
                if best_location:
                    self.resources -= 10
                    self.simulation.build_structure(best_location)
        except Exception as e:
            print(f"An error occurred while building a structure: {e}")
            self.simulation.terminal_output.append(f"An error occurred while building a structure: {e}")

    def trade(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Agent) and cell != self:
                    if cell.resources > self.resources:
                        trade_amount = (cell.resources - self.resources) // 2
                        cell.resources -= trade_amount
                        self.resources += trade_amount
        except Exception as e:
            print(f"An error occurred while trading: {e}")
            self.simulation.terminal_output.append(f"An error occurred while trading: {e}")

    def foraging(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Plant):
                    self.hunger += 10
                    self.energy += 10
                    break
        except Exception as e:
            print(f"An error occurred while foraging: {e}")
            self.simulation.terminal_output.append(f"An error occurred while foraging: {e}")

    def find_mate(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Agent) and cell.agent_type == self.agent_type:
                    self.social_need += 10
                    break
        except Exception as e:
            print(f"An error occurred while finding a mate: {e}")
            self.simulation.terminal_output.append(f"An error occurred while finding a mate: {e}")

    def hunt(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Agent) and cell.agent_type == 'prey':
                    self.hunger += 20
                    self.energy += 20
                    cell.alive = False
                    break
        except Exception as e:
            print(f"An error occurred while hunting: {e}")
            self.simulation.terminal_output.append(f"An error occurred while hunting: {e}")

    def rest(self):
        try:
            self.energy += 10
        except Exception as e:
            print(f"An error occurred while resting: {e}")
            self.simulation.terminal_output.append(f"An error occurred while resting: {e}")

    def explore(self):
        try:
            self.position = (random.randint(0, self.simulation.grid_size), random.randint(0, self.simulation.grid_size))
        except Exception as e:
            print(f"An error occurred while exploring: {e}")
            self.simulation.terminal_output.append(f"An error occurred while exploring: {e}")

    def social_interaction(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Agent) and cell.agent_type == self.agent_type:
                    self.social_need += 5
                    break
        except Exception as e:
            print(f"An error occurred while interacting socially: {e}")
            self.simulation.terminal_output.append(f"An error occurred while interacting socially: {e}")

    def mate(self):
        try:
            vision = self.perceive()
            for cell in vision:
                if isinstance(cell, Agent) and cell.agent_type == self.agent_type and cell.alive:
                    offspring = self.simulation.crossover(self, cell)
                    self.simulation.mutate(offspring)
                    self.simulation.population.append(offspring)
                    break
        except Exception as e:
            print(f"An error occurred while mating: {e}")
            self.simulation.terminal_output.append(f"An error occurred while mating: {e}")

    def perceive(self):
        try:
            vision = []
            for x in range(-self.vision_range, self.vision_range + 1):
                for y in range(-self.vision_range, self.vision_range + 1):
                    cell = self.simulation.get_cell(self.position[0] + x, self.position[1] + y)
                    vision.append(cell)
            return vision
        except Exception as e:
            print(f"An error occurred while perceiving the environment: {e}")
            self.simulation.terminal_output.append(f"An error occurred while perceiving the environment: {e}")
            return []

    def calculate_fitness(self):
        try:
            self.fitness = self.hunger + self.thirst + self.social_need + self.energy
            return self.fitness
        except Exception as e:
            print(f"An error occurred while calculating fitness: {e}")
            self.simulation.terminal_output.append(f"An error occurred while calculating fitness: {e}")
            return 0
