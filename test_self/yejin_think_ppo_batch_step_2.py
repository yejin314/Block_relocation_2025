import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import numpy as np
from environment import GridEnv  # 사용자 환경은 그대로 사용합니다.


# === ActorCritic 모델 (변경 없음) ===
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
        x = x.view(x.size(0), -1)  # reshape 대신 view를 사용하는 것이 일반적입니다.
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)


# === GAE 및 리턴 계산 함수 (부트스트래핑 로직 추가) ===
def compute_gae_and_returns(rewards, masks, values, last_value, gamma=0.99, lam=0.95):
    # last_value는 롤아웃의 마지막 상태에 대한 가치 추정치입니다.
    values = values + [last_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


# === PPO 학습 함수 (고정 롤아웃 및 가치 클리핑 적용) ===
def train_ppo_stable(env, model, optimizer, num_total_steps=50000,
                     rollout_length=256, gamma=0.99, lam=0.95,
                     eps_clip=0.2, K_epochs=4, batch_size=64,
                     value_clip_coef=0.5, entropy_coef=0.01, device="cpu"):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    all_rewards = []
    episode_rewards = []
    episode_num = 0

    state = env.reset()
    # num_total_steps 만큼 전체 학습을 진행합니다.
    for global_step in range(0, num_total_steps, rollout_length):
        # --- 1. 고정된 길이(rollout_length)만큼 데이터 수집 ---
        states, actions, log_probs, values, rewards, masks, old_values = [], [], [], [], [], [], []

        for step in range(rollout_length):
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())

            episode_rewards.append(reward)

            states.append(state_t)
            actions.append(action)
            log_probs.append(dist.log_prob(action))
            values.append(value.item())
            rewards.append(reward)
            masks.append(1 - int(done))
            old_values.append(value)  # 가치 클리핑을 위해 이전 가치 저장

            state = next_state
            if done:
                all_rewards.append(sum(episode_rewards))
                print(
                    f"[Step {global_step + step + 1}] Episode {episode_num + 1} finished. Reward: {sum(episode_rewards):.3f}, Relocations: {env.num_relocations}")
                episode_rewards = []
                episode_num += 1
                state = env.reset()

        # --- 2. GAE 및 리턴 계산 (부트스트래핑) ---
        # 롤아웃이 에피소드 중간에 끝났을 경우, 마지막 상태의 가치를 추정하여 계산에 사용합니다.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            last_state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value_tensor = model(last_state_t)
            last_value = last_value_tensor.item()

        returns = compute_gae_and_returns(rewards, masks, values, last_value, gamma, lam)

        # 텐서로 변환
        states = torch.cat(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()
        old_values = torch.cat(old_values).detach()
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # --- 3. 어드밴티지 계산 및 정규화 ---
        advantages = returns - old_values.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 4. PPO 업데이트 (K_epochs 동안 미니배치 학습) ---
        for _ in range(K_epochs):
            idxs = torch.randperm(rollout_length)
            for start in range(0, rollout_length, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits, value = model(states[mb_idx])
                    dist = Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[mb_idx])
                    entropy = dist.entropy().mean()

                    # Policy Loss (비율 및 클리핑)
                    ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
                    surr1 = ratio * advantages[mb_idx]
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[mb_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # --- ✨ Value Loss (클리핑 적용) ✨ ---
                    value = value.squeeze(-1)
                    value_clipped = old_values[mb_idx].squeeze(-1) + torch.clamp(
                        value - old_values[mb_idx].squeeze(-1), -eps_clip, eps_clip
                    )
                    loss_v_clipped = nn.MSELoss()(value_clipped, returns[mb_idx])
                    loss_v_unclipped = nn.MSELoss()(value, returns[mb_idx])
                    value_loss = value_clip_coef * torch.max(loss_v_clipped, loss_v_unclipped)

                    # 최종 손실
                    loss = policy_loss + value_loss - entropy_coef * entropy

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                # Gradient Clipping (학습 안정성 추가)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                scaler.step(optimizer)
                scaler.update()

    return model, all_rewards


# === 실행 ===
env = GridEnv(grid_rows=10, grid_cols=11)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = ActorCritic(env.observation_space.shape, env.action_space.n)
optimizer = optim.Adam(model.parameters(), lr=0.0003)  # 학습률을 약간 낮추는 것이 안정적인 경우가 많습니다.

# 학습 실행
model, all_rewards = train_ppo_stable(env, model, optimizer,
                                      num_total_steps=100000,  # 에피소드 대신 전체 스텝으로 관리
                                      rollout_length=512,
                                      batch_size=64,
                                      K_epochs=10,
                                      gamma=0.99,
                                      eps_clip=0.2,
                                      entropy_coef=0.01,  # 탐색 강도 조절
                                      device=device)

import os

file_path = '../재배치 불필요/'
os.makedirs(file_path, exist_ok=True)
reward_path = f'PPO_Stable_Rewards_{env.grid_rows}_{env.grid_cols}_1.png'

plt.plot(all_rewards)
plt.title("PPO Stable Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.savefig(file_path + reward_path)
plt.show()

# === 평가 모드 예시 (변경 없음) ===
state = env.reset()
done = False
while not done:
    env.render()  # 환경 시각화
    time.sleep(0.1)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = model(state_t)
        # 평가 시에는 가장 확률 높은 행동 선택
        action = torch.argmax(logits, dim=-1)
    state, _, done, _ = env.step(action.item())
print("Evaluation Finished.")