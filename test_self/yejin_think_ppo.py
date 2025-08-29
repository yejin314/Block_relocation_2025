import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from environment import GridEnv

# ----------- Actor-Critic 모델 -----------
class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(inplace=False)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False)
        )


        conv_out_size = 64 * input_shape[1] * input_shape[2]
        self.fc = nn.Linear(conv_out_size, 256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)

# ----------- GAE 계산 ----------- (변경 없음)
def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# ----------- PPO 학습 루프 (AMP 포함) -----------
def train_ppo(env, model, optimizer, num_episodes=1000, gamma=0.99, lam=0.95,
              eps_clip=0.2, K_epochs=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')

    all_rewards = []
    all_losses = []
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())
            values.append(value.item())
            rewards.append(reward)
            masks.append(1 - int(done))

            state = next_state

        returns = compute_gae(rewards, masks, values, gamma, lam)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values_t = torch.tensor(values, dtype=torch.float32).to(device)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)

        for _ in range(K_epochs):
            with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                logits, value = model(states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(value.squeeze(-1), returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        all_losses.append(loss.item())


        print(f"[Episode {episode+1}] Reward: {total_reward:.2f}, Loss: {loss.item():.4f}")

    return model, all_rewards, all_losses

# ----------- 실행 -----------
env = GridEnv(grid_rows=10, grid_cols=11)
input_shape = env.observation_space.shape
num_actions = env.action_space.n

model = ActorCritic(input_shape, num_actions)
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, all_rewards, all_losses = train_ppo(env, model, optimizer, num_episodes=1000, K_epochs=3)

if len(all_rewards) == 1000:
    plt.plot(all_rewards)
    plt.title("PPO Training Rewards")
    plt.savefig("PPO Training Rewards_3.png")
    plt.show()

    plt.plot(all_losses)
    plt.title("PPO Training Loss")
    plt.savefig("PPO Training Loss_3.png")
    plt.show()