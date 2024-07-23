import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.graphics import Color, Ellipse, Line
from kivy.clock import Clock
from threading import Thread
import random
from openai import OpenAI
from queue import Queue
import asyncio

torch.autograd.set_detect_anomaly(True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)
MODEL = "llama3-70b-8192"

BATCH_SIZE = 64
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
MEMORY_SIZE = 10000
NUM_NEURONS = 10
UPDATE_INTERVAL = 0.1

class IzhikevichNeuron(nn.Module):
    def __init__(self, a, b, c, d):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.reset()

    def reset(self):
        self.v = nn.Parameter(torch.tensor(-65.0))
        self.u = nn.Parameter(torch.tensor(self.b * -65.0))
        self.spike_history = [0] * 100
        self.adaptation = nn.Parameter(torch.tensor(0.0))

    def forward(self, I):
        batch_size = I.shape[0]
        v = self.v.expand(batch_size)
        u = self.u.expand(batch_size)
        adaptation = self.adaptation.expand(batch_size)

        spike = v >= 30
        v_next = torch.where(spike, torch.full_like(v, self.c), 
                        v + 0.04 * v**2 + 5 * v + 140 - u + I - adaptation)
        u_next = torch.where(spike, u + self.d, 
                        u + self.a * (self.b * v - u))
        
        adaptation_next = adaptation + 0.01 * (spike.float() - adaptation)
        
        self.v.data.copy_(v_next.mean().detach())
        self.u.data.copy_(u_next.mean().detach())
        self.adaptation.data.copy_(adaptation_next.mean().detach())
        
        self.spike_history.pop(0)
        self.spike_history.append(spike.float().mean().item())
        
        return v_next, spike.float()

class Network(nn.Module):
    def __init__(self, neuron_params, num_neurons):
        super().__init__()
        self.neurons = nn.ModuleList([IzhikevichNeuron(*neuron_params) for _ in range(num_neurons)])
        self.synapses = nn.Parameter(torch.rand(num_neurons, num_neurons) * 0.5)
        self.readout = nn.Linear(num_neurons, num_neurons)

    def forward(self, inputs):
        print(f"Input shape: {inputs.shape}")
        
        if len(inputs.shape) == 4:
            batch_size, _, num_neurons, time_steps = inputs.shape
            inputs = inputs.squeeze(1)
        elif len(inputs.shape) == 3:
            batch_size, num_neurons, time_steps = inputs.shape
        else:
            raise ValueError(f"Unexpected input shape: {inputs.shape}")
        
        spikes = torch.zeros(batch_size, time_steps, num_neurons, device=inputs.device)
        
        for t in range(time_steps):
            I = inputs[:, :, t]
            neuron_inputs = I + torch.mm(spikes[:, t-1] if t > 0 else torch.zeros_like(I), self.synapses)
            for i, neuron in enumerate(self.neurons):
                _, spike = neuron(neuron_inputs[:, i])
                spikes[:, t, i] = spike

        output = self.readout(spikes.mean(dim=1))
        print(f"Output shape: {output.shape}")
        return output

    def get_spike_rates(self):
        return [sum(n.spike_history) / len(n.spike_history) for n in self.neurons]

class ReinforcementLearningAgent:
    def __init__(self, network, lr=0.01, gamma=0.99, device='cpu'):
        self.network = network
        self.device = device
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = []
        self.epsilon = 1.0
        self.model_dir = 'saved_models'
        os.makedirs(self.model_dir, exist_ok=True)

    def select_action(self, state):
        print(f"State shape in select_action: {state.shape}")
        if np.random.rand() <= self.epsilon:
            return np.random.choice(state.shape[1])
        with torch.no_grad():
            q_values = self.network(state.unsqueeze(0))
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        print(f"State shape in store_transition: {state.shape}")
        self.memory.append((state.cpu(), action, reward, next_state.cpu(), done))
        if len(self.memory) > MEMORY_SIZE:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return 0

        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        print(f"States shape in replay: {states.shape}")
        print(f"Next states shape in replay: {next_states.shape}")

        current_q_values = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.network(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.functional.mse_loss(current_q_values, target_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        return loss.item()

    def save_model(self, epoch):
        torch.save(self.network.state_dict(), os.path.join(self.model_dir, f'network_epoch_{epoch}.pt'))

    async def generate_method_idea(self):
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant specializing in Python programming and neural network concepts."},
                {"role": "user", "content": "Suggest a unique idea for improving our neural network or reinforcement learning algorithm."}
            ]
            chat_completion = await asyncio.to_thread(client.chat.completions.create, messages=messages, model=MODEL)
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating method idea: {e}")
            return None

    async def generate_code(self, prompt):
        try:
            messages = [
                {"role": "system", "content": "You are an AI assistant specializing in Python programming and neural network implementation."},
                {"role": "user", "content": f"Create a Python function that implements the following idea: {prompt}. Provide only the function code without explanation."}
            ]
            chat_completion = await asyncio.to_thread(client.chat.completions.create, messages=messages, model=MODEL)
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

    def integrate_generated_code(self, code):
        try:
            local_vars = {}
            exec(code, globals(), local_vars)
            
            func_name = next((name for name, obj in local_vars.items() if callable(obj)), None)
            if func_name:
                func = local_vars[func_name]
                
                if hasattr(self.network, func_name):
                    setattr(self.network, func_name, func)
                else:
                    setattr(type(self.network), func_name, staticmethod(func))
                
                print(f"Successfully integrated new function: {func_name}")
                return True
            else:
                print("No callable function found in generated code.")
                return False
        except Exception as e:
            print(f"Error integrating generated code: {e}")
            return False

class NeuralNetworkVisualization(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.neurons = []
        self.connections = []

    def update(self, network):
        self.canvas.clear()
        with self.canvas:
            for i in range(NUM_NEURONS):
                for j in range(NUM_NEURONS):
                    strength = network.synapses[i][j].item()
                    if abs(strength) > 0.1:
                        start = ((i + 1) * self.width / (NUM_NEURONS + 1), self.height / 2)
                        end = ((j + 1) * self.width / (NUM_NEURONS + 1), self.height / 2)
                        Color(0, 1, 0, min(1, abs(strength)))
                        Line(points=[start[0], start[1], end[0], end[1]], width=1)

            spike_rates = network.get_spike_rates()
            for i, rate in enumerate(spike_rates):
                x = (i + 1) * self.width / (NUM_NEURONS + 1)
                y = self.height / 2
                size = 20 + rate * 20
                Color(1, 0, 0, min(1, rate + 0.2))
                Ellipse(pos=(x - size/2, y - size/2), size=(size, size))

class BrainSimulationApp(App):
    def __init__(self, update_queue, **kwargs):
        super().__init__(**kwargs)
        self.update_queue = update_queue

    def build(self):
        layout = BoxLayout(orientation='vertical')
        self.label = Label(text='Neural Network Visualization', size_hint=(1, 0.1))
        self.visualization = NeuralNetworkVisualization(size_hint=(1, 0.9))
        layout.add_widget(self.label)
        layout.add_widget(self.visualization)
        Clock.schedule_interval(self.update, 0.1)
        return layout

    def update(self, dt):
        try:
            while not self.update_queue.empty():
                network = self.update_queue.get_nowait()
                self.visualization.update(network)
        except Exception as e:
            print(f"Error updating visualization: {e}")

async def simulation_step(network, agent, state, epoch):
    print(f"State shape at start of simulation_step: {state.shape}")
    action = agent.select_action(state)
    reward = torch.randn(1, device=agent.device).item()
    next_state = torch.randn(1, NUM_NEURONS, 100, device=agent.device)
    done = False

    agent.store_transition(state, action, reward, next_state, done)
    loss = agent.replay()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Total Reward: {reward:.4f}, Loss: {loss:.4f}')
        agent.save_model(epoch)

    if epoch % 50 == 0:
        method_idea = await agent.generate_method_idea()
        if method_idea:
            print(f"Generated Method Idea: {method_idea}")
            generated_code = await agent.generate_code(method_idea)
            if generated_code:
                print(f"Generated Code:\n{generated_code}")
                if agent.integrate_generated_code(generated_code):
                    print("New functionality integrated into the network.")

    print(f"Next state shape at end of simulation_step: {next_state.shape}")
    return next_state, epoch + 1

async def simulation_loop(network, agent, state, update_queue):
    epoch = 0
    while True:
        state, epoch = await simulation_step(network, agent, state, epoch)
        update_queue.put(network)
        await asyncio.sleep(UPDATE_INTERVAL)

def run_app(update_queue):
    app = BrainSimulationApp(update_queue)
    app.run()

async def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    neuron_params = (0.02, 0.2, -65, 8)
    network = Network(neuron_params, NUM_NEURONS).to(device)
    agent = ReinforcementLearningAgent(network, device=device)

    update_queue = Queue()

    app_thread = Thread(target=run_app, args=(update_queue,))
    app_thread.start()

    state = torch.randn(1, NUM_NEURONS, 100, device=device)
    print(f"Initial state shape: {state.shape}")

    await simulation_loop(network, agent, state, update_queue)

if __name__ == '__main__':
    asyncio.run(main())
