# 파일 이름: environment.py

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque


class GridEnv(gym.Env):
    # ✨ 1. __init__ 메서드에 max_steps 파라미터 추가
    def __init__(self, grid_rows=10, grid_cols=12, max_steps=500):
        super().__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # 채널 0: 에이전트(타겟), 1: 방해물, 2: 출구
        self.num_channels = 3
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(self.num_channels, self.grid_rows, self.grid_cols),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)  # 0:상, 1:하, 2:좌, 3:우

        # ✨ 최대 스텝 수를 인스턴스 변수로 저장
        self.max_steps_per_episode = max_steps

        self.fig, self.ax = None, None
        self.reset()

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.done = False
        self.num_relocations = 0

        # ✨ 2. 에피소드 스텝 카운터 초기화
        self.current_step = 0

        self.agent_block = {'r1': 1, 'c1': 1, 'r2': 2, 'c2': 1}
        self.obstacle_blocks = [
            {'r1': 2, 'c1': 2, 'r2': 2, 'c2': 3},

        ]
        self.exit_area = {
            'r1': 0, 'c1': self.grid_cols - 1,
            'r2': self.grid_rows - 1, 'c2': self.grid_cols - 1
        }

        self._update_state()
        return self.state

    # ... ( _get_block_cells, _update_state, _is_overlap 등 다른 함수는 변경 없음) ...
    def _get_block_cells(self, block):
        if block is None: return []
        return [(r, c) for r in range(block['r1'], block['r2'] + 1)
                for c in range(block['c1'], block['c2'] + 1)]

    def _update_state(self):
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        for r, c in self._get_block_cells(self.agent_block): self.state[0, r, c] = 1
        for block in self.obstacle_blocks:
            for r, c in self._get_block_cells(block): self.state[1, r, c] = 1
        for r, c in self._get_block_cells(self.exit_area): self.state[2, r, c] = 1

    def _is_overlap(self, block1, block2):
        if block1 is None or block2 is None: return False
        return not (block1['c2'] < block2['c1'] or block1['c1'] > block2['c2'] or
                    block1['r2'] < block2['r1'] or block1['r1'] > block2['r2'])

    def _is_valid_position(self, block, ignore_block=None):
        if not (0 <= block['r1'] and block['r2'] < self.grid_rows and
                0 <= block['c1'] and block['c2'] < self.grid_cols):
            return False

        all_obstacles = self.obstacle_blocks + [self.agent_block]
        for other_block in all_obstacles:
            if other_block is not ignore_block and self._is_overlap(block, other_block):
                return False
        return True

    def _find_relocation_path_for_block(self, block_to_move):
        h = block_to_move['r2'] - block_to_move['r1']
        w = block_to_move['c2'] - block_to_move['c1']

        start_pos_key = (block_to_move['r1'], block_to_move['c1'])
        queue = deque([start_pos_key])
        visited = {start_pos_key}

        while queue:
            r1, c1 = queue.popleft()
            if (r1, c1) != start_pos_key:
                current_block = {'r1': r1, 'c1': c1, 'r2': r1 + h, 'c2': c1 + w}
                if self._is_valid_position(current_block, ignore_block=block_to_move):
                    return current_block

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_r1, next_c1 = r1 + dr, c1 + dc
                next_pos_key = (next_r1, next_c1)
                if next_pos_key in visited: continue

                next_block = {'r1': next_r1, 'c1': next_c1, 'r2': next_r1 + h, 'c2': next_c1 + w}
                if self._is_valid_position(next_block, ignore_block=block_to_move):
                    visited.add(next_pos_key)
                    queue.append(next_pos_key)
        return None

    def _is_move_valid_with_halo(self, block, direction):
        dr, dc = direction
        halo = {
            'r1': block['r1'] - 1, 'c1': block['c1'] - 1,
            'r2': block['r2'] + 1, 'c2': block['c2'] + 1
        }
        check_area = None
        if dr == -1:
            check_area = {'r1': halo['r1'], 'c1': halo['c1'], 'r2': halo['r1'], 'c2': halo['c2']}
        elif dr == 1:
            check_area = {'r1': halo['r2'], 'c1': halo['c1'], 'r2': halo['r2'], 'c2': halo['c2']}
        elif dc == -1:
            check_area = {'r1': halo['r1'], 'c1': halo['c1'], 'r2': halo['r2'], 'c2': halo['c1']}
        elif dc == 1:
            check_area = {'r1': halo['r1'], 'c1': halo['c2'], 'r2': halo['r2'], 'c2': halo['c2']}

        if check_area:
            for obs in self.obstacle_blocks:
                if self._is_overlap(check_area, obs):
                    return False
        return True

    def step(self, action):
        # 에피소드가 이미 종료된 경우 즉시 반환
        if self.done:
            return self.state, 0.0, True, {}

        # PPO 코드에 전달할 추가 정보 딕셔너리
        info = {}

        # 행동에 따른 방향 결정
        direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = direction_map[action]

        # 이동 후 예상 위치 계산
        next_agent_pos = {
            'r1': self.agent_block['r1'] + dr, 'c1': self.agent_block['c1'] + dc,
            'r2': self.agent_block['r2'] + dr, 'c2': self.agent_block['c2'] + dc
        }

        # ✨ 1. (버그 수정) 이동 위치가 그리드 경계를 벗어나는지 가장 먼저 확인
        if not (0 <= next_agent_pos['r1'] and next_agent_pos['r2'] < self.grid_rows and
                0 <= next_agent_pos['c1'] and next_agent_pos['c2'] < self.grid_cols):
            # 벽 충돌 시 페널티 없이(0.0) 현재 상태 그대로 반환
            return self.state, 0.0, self.done, info

        # ✨ 2. (기능 추가) 에피소드 스텝 카운터 업데이트 및 타임아웃 확인
        self.current_step += 1
        if self.current_step >= self.max_steps_per_episode:
            self.done = True
            info['termination_reason'] = 'timeout'

        # 기본적인 시간 경과 페널티
        reward = -0.1

        # 이동할 위치에 방해 블록이 있는지 확인
        collided_obstacle = None
        for i, obs in enumerate(self.obstacle_blocks):
            if self._is_overlap(next_agent_pos, obs):
                collided_obstacle = obs
                collided_obstacle_idx = i
                break

        # 분기: 장애물과 충돌했는가?
        if collided_obstacle:
            # 장애물 재배치 로직 수행
            new_pos = self._find_relocation_path_for_block(collided_obstacle)
            if new_pos:
                # 재배치 성공
                self.obstacle_blocks[collided_obstacle_idx] = new_pos
                self.num_relocations += 1
                reward -= 1.0
            else:
                # 재배치 실패 시 블록 제거
                self.obstacle_blocks.pop(collided_obstacle_idx)
                self.num_relocations += 2
                reward -= 2.0

            # 재배치 후 에이전트는 해당 위치로 이동
            self.agent_block = next_agent_pos

        # ✨ 3. (수정 사항) 밀 수 없는 장애물 옆(벽)으로 이동 시 페널티 없이 반환
        elif not self._is_move_valid_with_halo(self.agent_block, (dr, dc)):
            return self.state, 0.0, self.done, {}

        # 분기: 유효한 빈 공간으로 이동
        elif self._is_valid_position(next_agent_pos):
            self.agent_block = next_agent_pos

        # 분기: 그 외 모든 이동 불가 상황
        else:
            return self.state, 0.0, self.done, {}

        # ✨ 4. (기능 추가) 목표 지점 도달 시 성공 정보 기록
        if self.agent_block['c2'] == self.grid_cols - 1:
            reward += 1.0
            self.done = True
            info['termination_reason'] = 'success'

        # 최종 상태 업데이트 및 반환
        self._update_state()
        return self.state, reward, self.done, info

    def render(self, mode='human'):
        # ... (render 함수는 변경 없음) ...
        rgb_grid = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
        for r, c in self._get_block_cells(self.exit_area): rgb_grid[r, c] = [0, 255, 0]
        for block in self.obstacle_blocks:
            for r, c in self._get_block_cells(block): rgb_grid[r, c] = [128, 128, 128]
        for r, c in self._get_block_cells(self.agent_block): rgb_grid[r, c] = [255, 255, 0]

        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(7, 6))
            self.img = self.ax.imshow(rgb_grid)
            plt.ion();
            plt.show()
        else:
            self.img.set_data(rgb_grid)
            self.fig.canvas.draw();
            self.fig.canvas.flush_events()

        plt.pause(0.01)