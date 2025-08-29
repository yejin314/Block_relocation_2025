# 파일 이름: environment.py

import numpy as np
import gym
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque


class GridEnv(gym.Env):
    def __init__(self, grid_rows=10, grid_cols=12):
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

        self.fig, self.ax = None, None
        self.reset()

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.done = False
        self.num_relocations = 0  # 재배치 횟수 초기화

        self.agent_block = {'r1': 4, 'c1': 4, 'r2': 5, 'c2': 5}
        self.obstacle_blocks = [
            {'r1': 2, 'c1': 3, 'r2': 3, 'c2': 3},
            {'r1': 4, 'c1': 6, 'r2': 5, 'c2': 6},
            {'r1': 6, 'c1': 4, 'r2': 7, 'c2': 5}
        ]
        self.exit_area = {
            'r1': 0, 'c1': self.grid_cols - 1,
            'r2': self.grid_rows - 1, 'c2': self.grid_cols - 1
        }

        self._update_state()
        return self.state

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
        if self.done: return self.state, 0.0, True, {}

        direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = direction_map[action]

        next_agent_pos = {
            'r1': self.agent_block['r1'] + dr, 'c1': self.agent_block['c1'] + dc,
            'r2': self.agent_block['r2'] + dr, 'c2': self.agent_block['c2'] + dc
        }
        reward = -0.01

        # 이동할 위치에 방해 블록이 있는지 확인
        collided_obstacle = None
        for i, obs in enumerate(self.obstacle_blocks):
            if self._is_overlap(next_agent_pos, obs):
                collided_obstacle = obs
                collided_obstacle_idx = i
                break

        # 재배치 로직
        if collided_obstacle:
            new_pos = self._find_relocation_path_for_block(collided_obstacle)
            if new_pos:
                # 재배치 성공
                self.obstacle_blocks[collided_obstacle_idx] = new_pos
                self.num_relocations += 1
                reward -= 1.0
            else:
                # ✨ [핵심 수정] 재배치 실패 시 블록 제거 ✨
                self.obstacle_blocks.pop(collided_obstacle_idx)
                self.num_relocations += 2
                reward -= 2.0

            # 재배치 성공/실패 여부와 관계없이 에이전트는 이동
            self.agent_block = next_agent_pos

        # 방해 블록과 부딪히지 않은 경우 (기존 로직)
        elif not self._is_move_valid_with_halo(self.agent_block, (dr, dc)):
            return self.state, -0.1, self.done, {}

        elif self._is_valid_position(next_agent_pos):
            self.agent_block = next_agent_pos
        else:
            return self.state, -0.1, self.done, {}

        # 성공 조건
        if self.agent_block['c2'] == self.grid_cols - 1:
            reward += 10.0
            self.done = True

        self._update_state()
        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        # ... (이전 코드와 동일)
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