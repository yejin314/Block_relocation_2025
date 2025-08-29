# 파일 이름: main.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import time
import numpy as np
# ✨ 'environment.py'를 임포트하도록 수정
from environment_gemini_3 import GridEnv
from collections import deque
import pandas as pd
import os


# === ActorCritic 모델 및 GAE 함수 (변경 없음) ===
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
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.policy_head(x), self.value_head(x)


def compute_gae_and_returns(rewards, masks, values, last_value, gamma=0.99, lam=0.95):
    values = values + [last_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


# === PPO 학습 함수 (info 딕셔너리 사용하도록 수정) ===
def train_ppo_stable(env, model, optimizer, num_total_steps=50000,
                     rollout_length=256, gamma=0.99, lam=0.95,
                     eps_clip=0.2, K_epochs=4, batch_size=64,
                     value_clip_coef=0.5, entropy_coef=0.01, device="cpu"):
    print("--- 학습 시작 ---", flush=True)

    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    all_rewards, all_losses, all_relocations = [], [], []

    recent_rewards = deque(maxlen=20)
    recent_relocations = deque(maxlen=20)
    recent_successes = deque(maxlen=100)

    episode_rewards, episode_num, episode_len = [], 0, 0
    state = env.reset()

    for global_step in range(0, num_total_steps, rollout_length):
        success_rate = (np.mean(recent_successes) * 100) if recent_successes else 0
        print(
            f"[진행] Step {global_step}/{num_total_steps} | "
            f"최근 평균 보상: {np.mean(recent_rewards):.3f} | "
            f"최근 성공률: {success_rate:.1f}% | "
            f"최신 Loss: {all_losses[-1] if all_losses else 'N/A'}",
            flush=True
        )

        states, actions, log_probs, values, rewards, masks, old_values = [], [], [], [], [], [], []

        for step in range(rollout_length):
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits, value = model(state_t)
                dist = Categorical(logits=logits)
                action = dist.sample()

            next_state, reward, done, info = env.step(action.item())

            episode_rewards.append(reward)
            episode_len += 1

            states.append(state_t);
            actions.append(action);
            log_probs.append(dist.log_prob(action))
            values.append(value.item());
            rewards.append(reward);
            masks.append(1 - int(done))
            old_values.append(value)

            state = next_state
            if done:
                total_reward = sum(episode_rewards)
                all_rewards.append(total_reward)
                recent_rewards.append(total_reward)
                all_relocations.append(env.num_relocations)
                recent_relocations.append(env.num_relocations)

                # 새로운 환경은 info를 반환하지 않으므로, 성공 여부를 보상 기반으로 추정
                if episode_len < 100:  # 성공 보상이 양수일 경우
                    termination_log = "성공 (Success)"
                    recent_successes.append(1)
                else:  # 타임아웃 또는 실패
                    termination_log = "실패/타임아웃"
                    recent_successes.append(0)

                print(
                    f"[에피소드 종료] Ep {episode_num + 1} | 결과: {termination_log} | "
                    f"보상: {total_reward:.3f} | 길이: {episode_len} | 재배치: {env.num_relocations}", flush=True
                )
                episode_rewards, episode_len = [], 0
                episode_num += 1
                state = env.reset()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
            last_state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value_tensor = model(last_state_t)
            last_value = last_value_tensor.item()

        returns = compute_gae_and_returns(rewards, masks, values, last_value, gamma, lam)
        states, actions, old_log_probs = torch.cat(states), torch.stack(actions), torch.stack(log_probs).detach()
        old_values, returns = torch.cat(old_values).detach(), torch.tensor(returns, dtype=torch.float32, device=device)

        advantages = returns - old_values.squeeze(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        rollout_losses = []
        for _ in range(K_epochs):
            idxs = torch.randperm(rollout_length)
            for start_idx in range(0, rollout_length, batch_size):
                end_idx = start_idx + batch_size
                mb_idx = idxs[start_idx:end_idx]

                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    logits, value = model(states[mb_idx])
                    dist = Categorical(logits=logits)
                    new_log_probs, entropy = dist.log_prob(actions[mb_idx]), dist.entropy().mean()

                    ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
                    surr1 = ratio * advantages[mb_idx]
                    surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantages[mb_idx]
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value = value.squeeze(-1)
                    value_clipped = old_values[mb_idx].squeeze(-1) + torch.clamp(value - old_values[mb_idx].squeeze(-1),
                                                                                 -value_clip_coef, value_clip_coef)
                    loss_v_clipped = (value_clipped - returns[mb_idx]).pow(2)
                    loss_v_unclipped = (value - returns[mb_idx]).pow(2)
                    value_loss = 0.5 * torch.max(loss_v_clipped, loss_v_unclipped).mean()

                    loss = policy_loss + value_loss - entropy_coef * entropy

                rollout_losses.append(loss.item())
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # ✨ [수정] 안정성을 위해 grad clip 값 조정
                scaler.step(optimizer)
                scaler.update()

        all_losses.append(np.mean(rollout_losses))
    return model, all_rewards, all_losses, all_relocations


# === 실행 ===
if __name__ == '__main__':
    # ✨ max_steps는 환경에 맞게 조정 필요
    env = GridEnv(grid_rows=4, grid_cols=5)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- 환경 설정 ---\n사용 장치: {device}", flush=True)

    model = ActorCritic(env.observation_space.shape, env.action_space.n).to(device)
    # ✨ [수정] 학습률(lr)을 안정적인 값으로 조정
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    model, all_rewards, all_losses, all_relocations = train_ppo_stable(
        env, model, optimizer,
        num_total_steps=100000,
        rollout_length=512,
        batch_size=64,
        # ✨ [수정] 정책 업데이트 안정성을 위한 eps_clip 조정
        eps_clip=0.2,
        # ✨ [수정] 데이터 활용도를 높이기 위해 K_epochs 조정
        K_epochs=3,
        entropy_coef=0.01,
        device=device
    )

    file_path = './results/'
    os.makedirs(file_path, exist_ok=True)

    # --- 그래프 그리기 (2개의 그래프) ---
    # 1. 보상과 손실 그래프
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward', color='tab:blue')
    ax1.plot(all_rewards, color='tab:blue', label='Reward')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(all_losses, color='tab:red', alpha=0.6, label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    plt.title('PPO Training Rewards & Loss')
    reward_loss_path = os.path.join(file_path, f'PPO_Rewards_Loss_2.png')
    plt.savefig(reward_loss_path)
    plt.show()

    # ✨ [추가] 2. 재배치 횟수 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(all_relocations, color='tab:green')
    plt.title('Number of Relocations per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Relocation Count')
    plt.grid(True)
    relocations_path = os.path.join(file_path, f'PPO_Relocations_2.png')
    plt.savefig(relocations_path)
    plt.show()

    # === 평가 모드 ===
    state = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.1)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            logits, _ = model(state_t)
            action = torch.argmax(logits, dim=-1)
        state, _, done, _ = env.step(action.item())
    print("Evaluation Finished.", flush=True)