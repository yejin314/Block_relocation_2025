from environment import GridEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import pandas as pd

# === ActorCritic 모델 ===
class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=False)
        )
        conv_out_size = 64 * input_shape[1] * input_shape[2]
        self.fc = nn.Linear(conv_out_size, 256)
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)


# === GAE 계산 ===
def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step+1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


# === PPO 학습 함수 (AMP + 미니배치) ===
def train_ppo(env, model, optimizer, num_episodes=1000, gamma=0.99, lam=0.95,
              eps_clip=0.2, K_epochs=4, batch_size=64, device="cpu"):

    start = time.time()
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    all_rewards = []
    all_losses = []
    all_relocations = []

    for episode in range(num_episodes):
        # start = time.time()
        state = env.reset()
        done = False

        states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []

        # === Rollout 수집 ===
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=device).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device=="cuda")):
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())  # detach로 그래프 연결 차단
            values.append(value.item())
            rewards.append(reward)
            masks.append(1 - int(done))

            state = next_state

        # === Advantage 계산 ===
        returns = compute_gae(rewards, masks, values, gamma, lam)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.tensor(values, dtype=torch.float32, device=device)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)
        dataset_size = states.size(0)

        # === PPO 업데이트 (K_epochs 동안 미니배치 학습) ===
        for _ in range(K_epochs):
            idxs = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    logits, value = model(states[mb_idx])
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[mb_idx])
                    entropy = dist.entropy().mean()

                    ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
                    surr1 = ratio * advantages[mb_idx]
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[mb_idx]

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(value.squeeze(-1), returns[mb_idx])
                    loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)
        all_losses.append(loss.item())
        all_relocations.append(env.num_relocations)
        # end = time.time()

        print(f"[Episode {episode+1}] Reward: {total_reward:.3f}, Loss: {loss.item():.4f}, 재배치 횟수: {env.num_relocations}")


    return model, all_rewards, all_losses, all_relocations


# === 실행 ===
env = GridEnv(grid_rows=4, grid_cols=5)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = ActorCritic(env.observation_space.shape, env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model, all_rewards, all_losses, all_relocations = train_ppo(env, model, optimizer,
                                                           num_episodes=1000,
                                                           batch_size=64,
                                                           K_epochs=4,
                                                           gamma=0.99,
                                                           device=device)

import os

file_path = '../재배치 불필요/'
os.makedirs(file_path, exist_ok=True)
reward_path = f'PPO Training Rewards_AMP_{env.grid_rows}_{env.grid_cols}_3.png'
loss_path = f'PPO Training Loss_AMP_{env.grid_rows}_{env.grid_cols}_3.png'
relocation_path = f'PPO Training Relocation_AMP_{env.grid_rows}_{env.grid_cols}_3.png'
result_path = f'Result_{env.grid_rows}_{env.grid_cols}_3.xlsx'

plt.plot(all_rewards)
plt.title("PPO Training Rewards (AMP + MiniBatch)")
plt.savefig(file_path + reward_path)
plt.show()

plt.plot(all_losses)
plt.title("PPO Training Loss (AMP + MiniBatch)")
plt.savefig(file_path + loss_path)
plt.show()

plt.plot(all_relocations)
plt.title("PPO Training Relocation (AMP + MiniBatch)")
plt.savefig(file_path + relocation_path)
plt.show()

result_df = pd.DataFrame({'Episode': range(1, 1001), 'Reward': all_rewards, 'Loss': all_losses, 'Relocation': all_relocations})
result_df.to_excel(file_path + result_path, index=False)



# === 평가 모드 예시 ===
state = env.reset()
done = False
while not done:
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = model(state_t)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs, dim=-1)
    state, _, done, _ = env.step(action.item())
    print(state)
