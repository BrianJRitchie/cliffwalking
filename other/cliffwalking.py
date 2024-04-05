import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym

# Discount factor for future utilities
DISCOUNT_FACTOR = 0.99

# Number of episodes to run
NUM_EPISODES = 1000

# Max steps per episode
MAX_STEPS = 10000

# Score agent needs for environment to be solved
SOLVED_SCORE = -13  # Adjusted for CliffWalking-v0

# Device to run model on
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Define the Actor and Critic networks
class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.actor = nn.Linear(64, output_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.actor(x), self.critic(x)

# Initialize environment and model
env = gym.make('CliffWalking-v0')
input_dim = env.observation_space.n
output_dim = env.action_space.n
model = ActorCritic(input_dim, output_dim).to(DEVICE)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    log_probs = []
    values = []
    rewards = []

    for step in range(MAX_STEPS):
        state_tensor = torch.FloatTensor([state]).to(DEVICE)
        action_probs, value = model(state_tensor)
        action_probs = F.softmax(action_probs, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        next_state, reward, done, _ = env.step(action.item())

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)

        if done:
            break
        else:
            state = next_state

    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + DISCOUNT_FACTOR * R
        returns.insert(0, R)

    actor_loss = 0
    critic_loss = 0
    for log_prob, value, R in zip(log_probs, values, returns):
        advantage = R - value.item()
        actor_loss += -log_prob * advantage
        critic_loss += torch.pow(value - R, 2)

    optimizer.zero_grad()
    (actor_loss + critic_loss).backward()
    optimizer.step()

    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Actor Loss: {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

# Testing
state = env.reset()
total_reward = 0
while True:
    state_tensor = torch.FloatTensor([state]).to(DEVICE)
    action_probs, _ = model(state_tensor)
    action_probs = F.softmax(action_probs, dim=-1)
    action = torch.argmax(action_probs).item()
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()
    if done:
        break
    else:
        state = next_state

env.close()
