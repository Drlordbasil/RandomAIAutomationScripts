import os
import cv2
import gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from pytube import YouTube
from collections import deque
from torchvision import transforms

# Step 1: Extract Frames from YouTube Videos
def download_video(url, output_path='video.mp4'):
    yt = YouTube(url)
    stream = yt.streams.filter(file_extension='mp4').first()
    stream.download(filename=output_path)
    return output_path

def extract_frames(video_path, output_folder='frames', frame_skip=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = cap.read()
    while success:
        if frame_count % frame_skip == 0:
            cv2.imwrite(os.path.join(output_folder, f'frame_{frame_count}.jpg'), image)
        success, image = cap.read()
        frame_count += 1
    cap.release()

# Step 2: Frame Preprocessing
class FramePreprocessor:
    def __init__(self, frame_size=(84, 84)):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize(frame_size),
            transforms.ToTensor()
        ])

    def preprocess(self, frame):
        return self.transform(frame)

# Step 3: Define the Gym Environment
class VideoTileEnv(gym.Env):
    def __init__(self, frame_folder):
        super(VideoTileEnv, self).__init__()
        self.frame_files = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith('.jpg')])
        self.current_frame_index = 0
        self.action_space = spaces.Discrete(6)  # 6 possible movements
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 84, 84), dtype=np.float32)
        self.preprocessor = FramePreprocessor()
        self.safe_tiles = [(20, 20), (40, 40), (60, 60)]  # Example coordinates of safe tiles
        self.unsafe_tiles = [(30, 30), (50, 50), (70, 70)]  # Example coordinates of unsafe tiles

    def reset(self):
        self.current_frame_index = 0
        frame = self._get_current_frame()
        return frame, {}

    def _get_current_frame(self):
        frame = cv2.imread(self.frame_files[self.current_frame_index])
        frame = self.preprocessor.preprocess(frame)
        return frame

    def step(self, action):
        self.current_frame_index = min(self.current_frame_index + 1, len(self.frame_files) - 1)
        frame = self._get_current_frame()
        reward = self._calculate_reward()
        terminated = self.current_frame_index == len(self.frame_files) - 1
        truncated = False  # No early stopping in this simple environment
        return frame, reward, terminated, truncated, {}

    def _calculate_reward(self):
        # Example reward calculation based on tile safety
        if self.current_frame_index in self.safe_tiles:
            return 10  # Positive reward for safe tile
        elif self.current_frame_index in self.unsafe_tiles:
            return -10  # Negative reward for unsafe tile
        else:
            return -0.1  # Small penalty for each step to encourage faster goal-reaching

    def render(self, mode='human'):
        frame = cv2.imread(self.frame_files[self.current_frame_index])
        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

# Wrapper function to create the environment
def create_video_tile_env(frame_folder):
    return VideoTileEnv(frame_folder)

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)

def train_dqn(env, num_episodes, batch_size, gamma, epsilon_start, epsilon_end, epsilon_decay):
    replay_buffer = ReplayBuffer(10000)
    q_network = CNNQNetwork(env.observation_space.shape, env.action_space.n)
    target_network = CNNQNetwork(env.observation_space.shape, env.action_space.n)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters())
    loss_fn = nn.MSELoss()

    epsilon = epsilon_start
    for episode in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        for t in range(200):  # Use a fixed number of steps per episode
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = q_network(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).argmax().item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, terminated or truncated)
            state = next_state
            total_reward += reward
            
            if replay_buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_network(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)
                
                loss = loss_fn(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if terminated or truncated:
                break
        
        if episode % 10 == 0:
            target_network.load_state_dict(q_network.state_dict())
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Example usage
video_url = 'https://www.youtube.com/watch?v=qEJmEZmyG9w'
video_path = download_video(video_url)
extract_frames(video_path, frame_skip=30)

frame_folder = 'frames'
env = create_video_tile_env(frame_folder)
train_dqn(env, num_episodes=500, batch_size=64, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995)
