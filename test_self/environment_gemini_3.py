# 파일 이름: environment.py

import numpy as np
import gym
from gym import spaces
from collections import deque
import os


class GridEnv(gym.Env):
    def __init__(self, grid_rows=4, grid_cols=5):
        super(GridEnv, self).__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.grid_rows, self.grid_cols),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.yellow_block = {'row1': 1, 'col1': 0, 'row2': 2, 'col2': 1}
        self.gray_blocks = [
            {'row1': 1, 'col1': 2, 'row2': 2, 'col2': 2},

        ]
        self.done = False
        self.num_relocations = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((2, self.grid_rows, self.grid_cols), dtype=np.float32)
        y = self.yellow_block
        obs[0, y['row1']:y['row2'] + 1, y['col1']:y['col2'] + 1] = 1.0
        for block in self.gray_blocks:
            if block:
                obs[1, block['row1']:block['row2'] + 1, block['col1']:block['col2'] + 1] = 1.0
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {'success': True}

        info = {'success': False}
        direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = direction_map[action]
        nxt_block = {
            'row1': self.yellow_block['row1'] + dr, 'col1': self.yellow_block['col1'] + dc,
            'row2': self.yellow_block['row2'] + dr, 'col2': self.yellow_block['col2'] + dc
        }

        if not self._is_in_bounds(nxt_block):
            return self._get_obs(), -0.1, self.done, info

        # --- ✨ [추가] 가상 영역 로직 ---
        # 현재 블록 위치를 기준으로 가상 영역 생성
        vh = self._get_virtual_horizontal(self.yellow_block)
        vv = self._get_virtual_vertical(self.yellow_block)

        # 다음 위치가 가상 영역 밖이면 페널티 부여 후 턴 종료
        if not (self._is_overlap(nxt_block, vh) or self._is_overlap(nxt_block, vv)):
            return self._get_obs(), -0.1, self.done, info
        # --- 로직 추가 끝 ---

        colliding_idx = self._check_overlap(nxt_block)
        reward = -0.001

        if colliding_idx is not None:
            block_to_move = self.gray_blocks[colliding_idx]
            static_obstacles = [nxt_block] + [b for i, b in enumerate(self.gray_blocks) if i != colliding_idx and b]
            relocation_result = self._find_strategic_relocation(block_to_move, nxt_block, static_obstacles)

            if relocation_result:
                new_pos, _ = relocation_result
                self.gray_blocks[colliding_idx] = new_pos
                reward -= 1.0
                self.num_relocations += 1
                self.yellow_block = nxt_block
            else:
                self.gray_blocks[colliding_idx] = None
                reward -= 2.0
                self.num_relocations += 2
                self.yellow_block = nxt_block
        else:
            self.yellow_block = nxt_block

        if self.yellow_block['col2'] == self.grid_cols - 1:
            self.done = True
            reward += 0.0
            info['success'] = True

        return self._get_obs(), reward, self.done, info

    def render(self, mode='human'):
        grid = np.full((self.grid_rows, self.grid_cols), '⬜️')
        for block in self.gray_blocks:
            if block:
                grid[block['row1']:block['row2'] + 1, block['col1']:block['col2'] + 1] = '⬛️'
        y = self.yellow_block
        grid[y['row1']:y['row2'] + 1, y['col1']:y['col2'] + 1] = '🟨'
        os.system('cls' if os.name == 'nt' else 'clear')
        for row in grid:
            print(' '.join(row))
        print(f"Relocations: {self.num_relocations}")

    def _is_in_bounds(self, block):
        return (0 <= block['row1'] <= block['row2'] < self.grid_rows and
                0 <= block['col1'] <= block['col2'] < self.grid_cols)

    def _check_overlap(self, block):
        for idx, b in enumerate(self.gray_blocks):
            if b and not (block['col2'] < b['col1'] or block['col1'] > b['col2'] or
                          block['row2'] < b['row1'] or block['row1'] > b['row2']):
                return idx
        return None

    def _is_overlap(self, a, b):
        return not (a['col2'] < b['col1'] or a['col1'] > b['col2'] or
                    a['row2'] < b['row1'] or a['row1'] > b['row2'])

    # --- ✨ [추가] 가상 영역 헬퍼 함수 ---
    def _get_virtual_horizontal(self, block):
        return {'row1': max(0, block['row1'] - 1), 'col1': block['col1'],
                'row2': min(self.grid_rows - 1, block['row2'] + 1), 'col2': block['col2']}

    def _get_virtual_vertical(self, block):
        return {'row1': block['row1'], 'col1': max(0, block['col1'] - 1),
                'row2': block['row2'], 'col2': min(self.grid_cols - 1, block['col2'] + 1)}

    def _find_strategic_relocation(self, block_to_move, space_to_clear, static_obstacles):
        # (이전과 동일)
        queue = deque([(block_to_move, 0)])
        visited = set([(block_to_move['row1'], block_to_move['col1'])])
        static_occupied = set()
        for obs in static_obstacles:
            for r in range(obs['row1'], obs['row2'] + 1):
                for c in range(obs['col1'], obs['col2'] + 1):
                    static_occupied.add((r, c))
        candidate_locations = []
        while queue:
            current_block, moves = queue.popleft()
            if not self._is_overlap(current_block, space_to_clear):
                strategic_score = (self.grid_cols - 1 - current_block['col2']) + \
                                  min(current_block['row1'], self.grid_rows - 1 - current_block['row2'])
                final_score = moves - strategic_score * 0.5
                candidate_locations.append((final_score, current_block, moves))
            if moves > 5: continue
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                next_block_pos = {'row1': current_block['row1'] + dr, 'col1': current_block['col1'] + dc,
                                  'row2': current_block['row2'] + dr, 'col2': current_block['col2'] + dc}
                if (next_block_pos['row1'], next_block_pos['col1']) in visited: continue
                if not self._is_in_bounds(next_block_pos): continue
                next_occupied = set((r, c) for r in range(next_block_pos['row1'], next_block_pos['row2'] + 1)
                                    for c in range(next_block_pos['col1'], next_block_pos['col2'] + 1))
                if not next_occupied.isdisjoint(static_occupied): continue
                visited.add((next_block_pos['row1'], next_block_pos['col1']))
                queue.append((next_block_pos, moves + 1))
        if not candidate_locations: return None
        candidate_locations.sort(key=lambda x: x[0])
        best_candidate = candidate_locations[0]
        return best_candidate[1], best_candidate[2]