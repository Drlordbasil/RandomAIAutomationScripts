import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy_garden.graph import Graph, MeshLinePlot
from kivy.clock import Clock
from threading import Thread
import random
from groq import Groq

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

class IzhikevichNeuron(nn.Module):
    def __init__(self, a, b, c, d):
        super(IzhikevichNeuron, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v = -65
        self.u = b * self.v
        self.spike_history = [0]
        self.synaptic_strength = 0.5

    def forward(self, I):
        if self.v >= 30:  # Spike threshold
            self.v = self.c
            self.u = self.u + self.d  # Avoid in-place operation
            self.spike_history.append(1)
        else:
            self.spike_history.append(0)
        self.v = self.v + 0.04 * self.v**2 + 5 * self.v + 140 - self.u + I  # Avoid in-place operation
        self.u = self.u + self.a * (self.b * self.v - self.u)  # Avoid in-place operation
        return self.v

    def update_synaptic_strength(self, pre_spike, post_spike):
        if pre_spike and post_spike:
            self.synaptic_strength += 0.005  # Strengthen the synapse
        elif pre_spike and not post_spike:
            self.synaptic_strength -= 0.003  # Weaken the synapse
        self.synaptic_strength = max(0, min(1, self.synaptic_strength))  # Clamp between 0 and 1

class Network(nn.Module):
    def __init__(self, neuron_params, num_neurons):
        super(Network, self).__init__()
        self.neurons = nn.ModuleList([IzhikevichNeuron(*neuron_params) for _ in range(num_neurons)])
        self.synapses = nn.Parameter(torch.rand(num_neurons, num_neurons) * 0.5)

    def forward(self, inputs):
        batch_size, num_neurons = inputs.size()
        outputs = torch.zeros(batch_size, len(self.neurons))
        for t in range(batch_size):
            I = inputs[t, :]
            for i, neuron in enumerate(self.neurons):
                input_current = I[i] + torch.sum(self.synapses[:, i] * torch.tensor([n.spike_history[-1] for n in self.neurons], dtype=torch.float32))
                outputs[t, i] = neuron(input_current)
        return outputs

class AdvancedReinforcementLearningAgent:
    def __init__(self, network, lr=0.01, gamma=0.99):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma  # Discount factor
        self.memory = []  # Experience replay buffer
        self.batch_size = 64
        self.epsilon = 1.0  # Exploration factor
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Create a directory for saving models
        self.model_dir = 'saved_models'
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def select_action(self, state):
        try:
            if np.random.rand() <= self.epsilon:
                return np.random.choice(len(state))
            with torch.no_grad():
                state = state.unsqueeze(0)  # Add batch dimension
                q_values = self.network(state)
            return q_values.argmax().item()
        except Exception as e:
            print(f"Error selecting action: {e}")

    def store_transition(self, state, action, reward, next_state, done):
        try:
            self.memory.append((state, action, reward, next_state, done))
            if len(self.memory) > 10000:
                self.memory.pop(0)
        except Exception as e:
            print(f"Error storing transition: {e}")

    def replay(self):
        try:
            if len(self.memory) < self.batch_size:
                return
            minibatch = random.sample(self.memory, self.batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target += self.gamma * torch.max(self.network(next_state.unsqueeze(0)))
                current_q_values = self.network(state.unsqueeze(0))
                target_q_values = current_q_values.clone()
                target_q_values[0, action] = target
                loss = nn.functional.mse_loss(current_q_values, target_q_values)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)  # Retain the graph for multiple backward passes
                self.optimizer.step()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        except Exception as e:
            print(f"Error during replay: {e}")

    def save_model(self, epoch):
        try:
            model_path = os.path.join(self.model_dir, f'network_epoch_{epoch}.pt')
            torch.save(self.network.state_dict(), model_path)
            print(f"Model saved at {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def generate_method_idea(self):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Give me an idea for a Python function."}
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating method idea: {e}")
            return None

    def generate_code(self, prompt):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating code: {e}")
            return None

    def evaluate_code(self, code):
        try:
            exec_locals = {}
            exec(code, {}, exec_locals)
            return exec_locals
        except SyntaxError as e:
            print(f"Syntax error in code: {e}")
            return None
        except Exception as e:
            print(f"Error evaluating code: {e}")
            return None

    def get_feedback(self, code, result):
        try:
            feedback_prompt = f"Evaluate the following Python code and its execution result:\n\nCode:\n{code}\n\nResult:\n{result}\n\nProvide feedback and a grade (0 to 100) for the code quality and correctness."
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": feedback_prompt}
                ],
                model="llama3-8b-8192",
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error getting feedback: {e}")
            return None

class BrainSimulationApp(App):
    def build(self):
        self.graph = Graph(xlabel='Time Step', ylabel='Membrane Potential (mV)',
                           x_ticks_minor=5, x_ticks_major=25, y_ticks_major=1,
                           y_grid_label=True, x_grid_label=True, padding=5,
                           x_grid=True, y_grid=True, xmin=0, xmax=100, ymin=-80, ymax=30)

        self.plot = MeshLinePlot(color=[1, 0, 0, 1])
        self.graph.add_plot(self.plot)
        self.synapse_plot = MeshLinePlot(color=[0, 1, 0, 1])
        self.graph.add_plot(self.synapse_plot)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Label(text='Neuron Membrane Potentials and Synaptic Strengths'))
        layout.add_widget(self.graph)
        return layout

    def update_plot(self, neuron_data, synapse_data):
        self.plot.points = [(i, v) for i, v in enumerate(neuron_data)]
        self.synapse_plot.points = [(i, v) for i, v in enumerate(synapse_data)]

def run_simulation_step(network, agent, state, neuron_data, synapse_data, app, epoch):
    try:
        action = agent.select_action(state[0, :])
        reward = torch.randn(1).item()  # Simulate reward
        next_state = state.clone()
        done = False  # End of episode condition

        agent.store_transition(state[0, :], action, reward, next_state[0, :], done)
        agent.replay()

        # Collect neuron membrane potentials for visualization
        with torch.no_grad():
            outputs = network(state)
        neuron_data.append(outputs[0, 0].item())  # Collect data for the first neuron

        # Update synaptic strengths for visualization
        for neuron in network.neurons:
            neuron.update_synaptic_strength(neuron.spike_history[-2], neuron.spike_history[-1])
            synapse_data.append(neuron.synaptic_strength)

        # Schedule plot update on the main thread
        if len(neuron_data) > 100:
            neuron_data.pop(0)
        if len(synapse_data) > 100:
            synapse_data.pop(0)
        Clock.schedule_once(lambda dt: app.update_plot(neuron_data, synapse_data), 0)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Total Reward: {reward}')
            agent.save_model(epoch)

        # Generate and evaluate new code every 50 epochs
        if epoch % 50 == 0:
            method_idea = agent.generate_method_idea()
            if method_idea:
                prompt = f"Create a Python function that {method_idea}."
                generated_code = agent.generate_code(prompt)
                if generated_code:
                    print(f"Generated Code:\n{generated_code}")
                    # Extract code from the response
                    code_start = generated_code.find("```python")
                    code_end = generated_code.find("```", code_start + 3)
                    if code_start != -1 and code_end != -1:
                        code_to_evaluate = generated_code[code_start + 9:code_end].strip()
                        evaluation_result = agent.evaluate_code(code_to_evaluate)
                        if evaluation_result:
                            print(f"Evaluation Result: {evaluation_result}")
                            feedback = agent.get_feedback(code_to_evaluate, evaluation_result)
                            print(f"Feedback:\n{feedback}")

                            # Use feedback to adjust training
                            if "grade" in feedback:
                                grade = int(feedback.split("grade:")[1].strip().split()[0])
                                if grade > 75:
                                    agent.epsilon *= 0.9  # Reward: Decrease exploration
                                else:
                                    agent.epsilon *= 1.1  # Punishment: Increase exploration

        # Schedule the next simulation step
        Clock.schedule_once(lambda dt: run_simulation_step(network, agent, next_state, neuron_data, synapse_data, app, epoch + 1), 1)

    except Exception as e:
        print(f"Error during simulation step: {e}")

if __name__ == '__main__':
    neuron_params = (0.02, 0.2, -65, 8)
    num_neurons = 10
    steps = 100

    # Initialize network and agent
    network = Network(neuron_params, num_neurons)
    agent = AdvancedReinforcementLearningAgent(network)

    # Initialize Kivy app
    app = BrainSimulationApp()
    app_thread = Thread(target=app.run)
    app_thread.start()

    # Initial state
    state = torch.randn(1, num_neurons)  # Simulate initial state
    neuron_data = []
    synapse_data = []

    # Run simulation steps with periodic updates
    Clock.schedule_once(lambda dt: run_simulation_step(network, agent, state, neuron_data, synapse_data, app, 0), 1)
