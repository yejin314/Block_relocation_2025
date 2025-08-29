from yejin_think_ppo import GridEnv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

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


# === 행동 선택 함수 ===
def select_action(state, model, deterministic=False, device="cpu"):
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = model(state_t)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.probs, dim=-1)
        else:
            action = dist.sample()
    return action.item(), dist.log_prob(action)


# === PPO 학습 ===
def train_ppo(env, model, optimizer, num_episodes=1000, gamma=0.99, lam=0.95,
              eps_clip=0.2, K_epochs=4, device="cpu"):

    model.to(device)
    all_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        states, actions, log_probs, values, rewards, masks = [], [], [], [], [], []

        # --- Rollout 데이터 수집 ---
        while not done:
            state_t = torch.tensor(state, dtype=torch.float32,
                                   device=device).unsqueeze(0)

            with torch.no_grad():  # 메모리 절약
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            # detach / item 처리로 그래프 연결 제거
            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action).detach())
            values.append(value.item())
            rewards.append(reward)
            masks.append(1 - int(done))

            state = next_state

        # --- Advantage 계산 ---
        returns = compute_gae(rewards, masks, values, gamma, lam)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        values_t = torch.tensor(values, dtype=torch.float32, device=device)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs)

        # --- PPO 업데이트 ---
        for _ in range(K_epochs):
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
            loss.backward()
            optimizer.step()

        total_reward = sum(rewards)
        all_rewards.append(total_reward)

        print(f"[Episode {episode+1}] Reward: {total_reward:.2f}, Loss: {loss.item():.4f}")

    return model, all_rewards


# === 실행 예시 ===
env = GridEnv()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ActorCritic(env.observation_space.shape, env.action_space.n).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

model, all_rewards = train_ppo(env, model, optimizer, num_episodes=200, device=device, K_epochs=2)

plt.plot(all_rewards)
plt.title("PPO Training Rewards")
plt.show()

# 평가 모드
state = env.reset()
done = False
while not done:
    action, _ = select_action(state, model, deterministic=True, device=device)
    state, _, done, _ = env.step(action)
