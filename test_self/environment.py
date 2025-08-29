import numpy as np
import gym
from gym import spaces

class GridEnv(gym.Env):
    def __init__(self, grid_rows=4, grid_cols=5):
        super(GridEnv, self).__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.grid_rows, self.grid_cols),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 상/하/좌/우
        self.reset()

    def reset(self):
        self.yellow_block = {'row1': 1, 'col1': 1, 'row2': 2, 'col2': 1}  # 이동 블록

        self.gray_blocks = [
            {'row1': 2, 'col1': 2, 'row2': 2, 'col2': 3},  # (0,2)-(2,2)

        ]
        self.done = False
        self.num_relocations = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((2, self.grid_rows, self.grid_cols), dtype=np.float32)
        for r in range(self.yellow_block['row1'], self.yellow_block['row2'] + 1):
            for c in range(self.yellow_block['col1'], self.yellow_block['col2'] + 1):
                obs[0, r, c] = 1.0
        for block in self.gray_blocks:
            if block is None:
                continue
            for r in range(block['row1'], block['row2'] + 1):
                for c in range(block['col1'], block['col2'] + 1):
                    obs[1, r, c] = 1.0
        return obs

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, {}

        direction_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        dr, dc = direction_map[action]
        nxt_block = {
            'row1': self.yellow_block['row1'] + dr,
            'col1': self.yellow_block['col1'] + dc,
            'row2': self.yellow_block['row2'] + dr,
            'col2': self.yellow_block['col2'] + dc
        }

        if not self._is_in_bounds(nxt_block):
            return self._get_obs(), -1.0, self.done, {}

        overlap_idx = self._check_overlap(nxt_block)
        reward = -0.001

        if overlap_idx is not None:
            occupied_cells = self._build_occupancy(exclude_idx=overlap_idx)
            vh = self._get_virtual_horizontal(nxt_block)
            vv = self._get_virtual_vertical(nxt_block)
            # 회색 블록 재배치 시 노란 블록도 장애물로 포함하여 경로 검증 추가
            other_blocks = [b for idx, b in enumerate(self.gray_blocks) if idx != overlap_idx and b is not None] + [nxt_block]
            new_pos = self._find_relocation_position(self.gray_blocks[overlap_idx], occupied_cells, vh, vv, other_blocks)
            if new_pos:
                self.gray_blocks[overlap_idx] = new_pos
                self.num_relocations += 1
                reward -= 1.0
                self.yellow_block = nxt_block
            else:
                self.gray_blocks[overlap_idx] = None
                self.num_relocations += 2
                reward -= 2.0
                self.yellow_block = nxt_block
        else:
            self.yellow_block = nxt_block

        if self.yellow_block['col2'] == self.grid_cols - 1:
            self.done = True
            reward += 0.0

        return self._get_obs(), reward, self.done, {}

    # --- 유틸 함수들 ---

    def _is_in_bounds(self, block):
        return 0 <= block['row1'] <= block['row2'] < self.grid_rows and \
               0 <= block['col1'] <= block['col2'] < self.grid_cols

    def _check_overlap(self, block):
        for idx, b in enumerate(self.gray_blocks):
            if b is None:
                continue
            if not (block['col2'] < b['col1'] or block['col1'] > b['col2'] or
                    block['row2'] < b['row1'] or block['row1'] > b['row2']):
                return idx
        return None

    def _build_occupancy(self, exclude_idx=None):
        occupied = set()
        for idx, b in enumerate(self.gray_blocks):
            if idx == exclude_idx or b is None:
                continue
            for r in range(b['row1'], b['row2'] + 1):
                for c in range(b['col1'], b['col2'] + 1):
                    occupied.add((r, c))
        return occupied

    def _get_virtual_horizontal(self, block):
        return {
            'row1': block['row1'],
            'col1': max(0, block['col1'] - 1),
            'row2': block['row2'],
            'col2': min(self.grid_cols - 1, block['col2'] + 1)
        }

    def _get_virtual_vertical(self, block):
        return {
            'row1': max(0, block['row1'] - 1),
            'col1': block['col1'],
            'row2': min(self.grid_rows - 1, block['row2'] + 1),
            'col2': block['col2']
        }

    def _is_overlap(self, a, b):
        return not (a['col2'] < b['col1'] or a['col1'] > b['col2'] or
                    a['row2'] < b['row1'] or a['row1'] > b['row2'])

    def _find_relocation_position(self, block, occupied_cells, vh, vv, other_blocks):
        """
        재배치 위치 탐색 (경로검증 포함)
        block: 재배치 대상 블록
        occupied_cells: 점유 집합 (자신 제외)
        vh, vv: 가상 금지영역
        other_blocks: 장애물로 참고할 블록 리스트 (회색 및 노란 블록 포함)
        """
        from collections import deque

        def can_move_block(start_block, target_block, obstacles):
            """경로 이동 가능 여부 BFS 검사"""
            visited = set()
            queue = deque([start_block])
            occupied = set()
            for obs in obstacles:
                for r in range(obs['row1'], obs['row2'] + 1):
                    for c in range(obs['col1'], obs['col2'] + 1):
                        occupied.add((r, c))
            while queue:
                cur = queue.popleft()
                if (cur['row1'], cur['col1']) == (target_block['row1'], target_block['col1']):
                    return True
                if (cur['row1'], cur['col1']) in visited:
                    continue
                visited.add((cur['row1'], cur['col1']))
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = {
                        'row1': cur['row1'] + dr,
                        'col1': cur['col1'] + dc,
                        'row2': cur['row2'] + dr,
                        'col2': cur['col2'] + dc
                    }
                    if not self._is_in_bounds(nxt):
                        continue
                    # 각 칸 위치 확인
                    cells = [(r, c) for r in range(nxt['row1'], nxt['row2'] + 1) for c in range(nxt['col1'], nxt['col2'] + 1)]
                    if any(cell in occupied for cell in cells):
                        continue
                    queue.append(nxt)
            return False

        h = block['row2'] - block['row1'] + 1
        w = block['col2'] - block['col1'] + 1
        for r in range(self.grid_rows - h + 1):
            for c in range(self.grid_cols - w + 1):
                cand = {'row1': r, 'col1': c, 'row2': r + h - 1, 'col2': c + w - 1}
                if self._is_overlap(cand, vh) or self._is_overlap(cand, vv):
                    continue
                cells = [(rr, cc) for rr in range(cand['row1'], cand['row2'] + 1)
                                      for cc in range(cand['col1'], cand['col2'] + 1)]
                if any(cell in occupied_cells for cell in cells):
                    continue
                # 경로 검증 - 실제 이동 가능한 위치인지 검사
                if can_move_block(block, cand, other_blocks):
                    return cand
        return None
