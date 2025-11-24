import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNQNetwork, self).__init__()
        # input_shape is (C, H, W) -> (4, 84, 84)
        # Enhanced architecture for bullet hell: More layers to detect complex patterns
        # Layer 1: Detects simple edges and corners (e.g., the edge of a bullet).
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            # Layer 2: Combines edges into shapes (e.g., a square bullet, the player ship).
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # Layer 3: Detects spatial relationships (e.g., "bullet approaching player").
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            # Layer 4: Detects complex multi-bullet patterns and trajectories
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate FC input size
        # 84x84 -> conv1(k=8,s=4) -> 20x20
        # 20x20 -> conv2(k=4,s=2) -> 9x9
        # 9x9 -> conv3(k=3,s=1) -> 7x7
        # 7x7 -> conv4(k=3,s=1) -> 5x5
        # So 128 * 5 * 5 = 3200
        self.fc_input_dim = 128 * 5 * 5
        
        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity, input_shape=(4, 84, 84)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate memory to avoid constant allocation/copying
        self.states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *input_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.uint8)

    def push(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = int(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )

    def __len__(self):
        return self.size

    def serialize(self):
        """Return a serializable snapshot of the buffer."""
        return {
            "states": self.states[:self.size].copy(),
            "actions": self.actions[:self.size].copy(),
            "rewards": self.rewards[:self.size].copy(),
            "next_states": self.next_states[:self.size].copy(),
            "dones": self.dones[:self.size].copy(),
            "ptr": self.ptr,
            "size": self.size,
            "capacity": self.capacity,
        }

    def load_from(self, data):
        """Restore buffer contents from a serialized snapshot."""
        size = min(data.get("size", 0), self.capacity)
        if size == 0:
            self.ptr = 0
            self.size = 0
            return
        self.states[:size] = data["states"][:size]
        self.actions[:size] = data["actions"][:size]
        self.rewards[:size] = data["rewards"][:size]
        self.next_states[:size] = data["next_states"][:size]
        self.dones[:size] = data["dones"][:size]
        self.size = size
        # ptr can be beyond size; wrap safely
        self.ptr = data.get("ptr", size) % self.capacity

class DQNAgent:
    def __init__(
        self,
        input_shape,
        num_actions,
        lr=1e-4,
        gamma=0.99,
        buffer_size=10000,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        double_dqn=False
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        self.double_dqn = double_dqn

        self.policy_net = CNNQNetwork(input_shape, num_actions).to(device)
        self.target_net = CNNQNetwork(input_shape, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size, input_shape=input_shape)

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(self.num_actions)
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device) / 255.0
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        state_t = torch.FloatTensor(state).to(self.device) / 255.0
        next_state_t = torch.FloatTensor(next_state).to(self.device) / 255.0
        action_t = torch.LongTensor(action).to(self.device).unsqueeze(1)
        reward_t = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        done_t = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        # Current Q values
        curr_q = self.policy_net(state_t).gather(1, action_t)

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Select action with policy net, evaluate with target net
                next_actions = self.policy_net(next_state_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_state_t).gather(1, next_actions)
            else:
                # Vanilla DQN: Max over target net
                next_q = self.target_net(next_state_t).max(1)[0].unsqueeze(1)
            
            target_q = reward_t + (1 - done_t) * self.gamma * next_q

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_full_state(self):
        """Return full training state for checkpointing."""
        return {
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "replay": self.memory.serialize(),
        }

    def load_full_state(self, state_dict):
        """Load full training state from checkpoint."""
        self.policy_net.load_state_dict(state_dict["policy_state"])
        self.target_net.load_state_dict(state_dict["target_state"])
        self.optimizer.load_state_dict(state_dict["optimizer_state"])
        replay = state_dict.get("replay", {})
        if replay:
            self.memory.load_from(replay)
