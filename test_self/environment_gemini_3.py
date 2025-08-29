import numpy as np
import gym
from gym import spaces
from collections import deque


class GridEnv(gym.Env):
    def __init__(self, grid_rows=4, grid_cols=5):
        super(GridEnv, self).__init__()
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        # 채널 0: 노란 블록(에이전트), 채널 1: 회색 블록(장애물)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(2, self.grid_rows, self.grid_cols),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(4)  # 상/하/좌/우
        self.reset()

    def reset(self):
        self.yellow_block = {'row1': 1, 'col1': 0, 'row2': 2, 'col2': 1}  # 이동 블록
        self.gray_blocks = [
            {'row1': 1, 'col1': 2, 'row2': 2, 'col2': 2},
        ]
        self.done = False
        self.num_relocations = 0
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((2, self.grid_rows, self.grid_cols), dtype=np.float32)
        # 노란 블록 그리기
        for r in range(self.yellow_block['row1'], self.yellow_block['row2'] + 1):
            for c in range(self.yellow_block['col1'], self.yellow_block['col2'] + 1):
                obs[0, r, c] = 1.0
        # 회색 블록들 그리기
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

        # 1. 경계 확인 (벽에 부딪히면 -1.0 페널티)
        if not self._is_in_bounds(nxt_block):
            return self._get_obs(), 0.0, self.done, {}

        # 2. 장애물 충돌 확인
        overlap_idx = self._check_overlap(nxt_block)
        reward = -0.001  # 기본 시간 페널티

        # 3. 충돌 시 재배치 로직 실행
        if overlap_idx is not None:
            occupied_cells = self._build_occupancy(exclude_idx=overlap_idx)
            vh = self._get_virtual_horizontal(nxt_block)
            vv = self._get_virtual_vertical(nxt_block)

            # 재배치 경로 탐색 시, 이동 후의 노란 블록도 장애물로 간주
            other_blocks = [b for idx, b in enumerate(self.gray_blocks) if idx != overlap_idx and b is not None] + [
                nxt_block]
            new_pos = self._find_relocation_position(self.gray_blocks[overlap_idx], occupied_cells, vh, vv,
                                                     other_blocks)

            if new_pos:  # 재배치 성공
                self.gray_blocks[overlap_idx] = new_pos
                self.num_relocations += 1
                reward -= 1.0
                self.yellow_block = nxt_block
            else:  # 재배치 실패 (블록 제거)
                self.gray_blocks[overlap_idx] = None
                self.num_relocations += 2
                reward -= 2.0
                self.yellow_block = nxt_block
        else:  # 충돌하지 않은 경우
            self.yellow_block = nxt_block

        # 4. 종료 조건 확인 (오른쪽 끝 도달)
        if self.yellow_block['col2'] == self.grid_cols - 1:
            self.done = True
            reward += 0.0  # 성공 보상

        return self._get_obs(), reward, self.done, {}

    # --- 유틸리티 함수들 ---

    def _is_in_bounds(self, block):
        return 0 <= block['row1'] <= block['row2'] < self.grid_rows and \
            0 <= block['col1'] <= block['col2'] < self.grid_cols

    def _check_overlap(self, block):
        for idx, b in enumerate(self.gray_blocks):
            if b is None: continue
            if not (block['col2'] < b['col1'] or block['col1'] > b['col2'] or
                    block['row2'] < b['row1'] or block['row1'] > b['row2']):
                return idx
        return None

    def _build_occupancy(self, exclude_idx=None):
        occupied = set()
        for idx, b in enumerate(self.gray_blocks):
            if idx == exclude_idx or b is None: continue
            for r in range(b['row1'], b['row2'] + 1):
                for c in range(b['col1'], b['col2'] + 1):
                    occupied.add((r, c))
        return occupied

    def _get_virtual_horizontal(self, block):
        return {'row1': block['row1'], 'col1': max(0, block['col1'] - 1),
                'row2': block['row2'], 'col2': min(self.grid_cols - 1, block['col2'] + 1)}

    def _get_virtual_vertical(self, block):
        return {'row1': max(0, block['row1'] - 1), 'col1': block['col1'],
                'row2': min(self.grid_rows - 1, block['row2'] + 1), 'col2': block['col2']}

    def _is_overlap(self, a, b):
        return not (a['col2'] < b['col1'] or a['col1'] > b['col2'] or
                    a['row2'] < b['row1'] or a['row1'] > b['row2'])

    def _find_relocation_position(self, block, occupied_cells, vh, vv, other_blocks):
        from collections import deque

        def can_move_block(start_block, target_block, obstacles):
            visited = set()
            start_key = (start_block['row1'], start_block['col1'])
            queue = deque([start_key])
            visited.add(start_key)

            # 경로 탐색 시 장애물이 차지하는 모든 셀 집합
            obstacle_cells = set()
            for obs in obstacles:
                for r_ in range(obs['row1'], obs['row2'] + 1):
                    for c_ in range(obs['col1'], obs['col2'] + 1):
                        obstacle_cells.add((r_, c_))

            h_ = start_block['row2'] - start_block['row1']
            w_ = start_block['col2'] - start_block['col1']

            while queue:
                r1, c1 = queue.popleft()
                if (r1, c1) == (target_block['row1'], target_block['col1']):
                    return True

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_r1, next_c1 = r1 + dr, c1 + dc
                    next_key = (next_r1, next_c1)

                    if next_key in visited: continue

                    next_block = {'row1': next_r1, 'col1': next_c1, 'row2': next_r1 + h_, 'col2': next_c1 + w_}

                    if not self._is_in_bounds(next_block): continue

                    # 이동할 위치가 다른 장애물과 겹치는지 확인
                    is_blocked = False
                    for r_ in range(next_block['row1'], next_block['row2'] + 1):
                        for c_ in range(next_block['col1'], next_block['col2'] + 1):
                            if (r_, c_) in obstacle_cells:
                                is_blocked = True
                                break
                        if is_blocked: break

                    if not is_blocked:
                        visited.add(next_key)
                        queue.append(next_key)
            return False

        h = block['row2'] - block['row1'] + 1
        w = block['col2'] - block['col1'] + 1

        for r in range(self.grid_rows - h + 1):
            for c in range(self.grid_cols - w + 1):
                cand = {'row1': r, 'col1': c, 'row2': r + h - 1, 'col2': c + w - 1}
                if self._is_overlap(cand, vh) or self._is_overlap(cand, vv): continue

                cells = [(rr, cc) for rr in range(cand['row1'], cand['row2'] + 1) for cc in
                         range(cand['col1'], cand['col2'] + 1)]
                if any(cell in occupied_cells for cell in cells): continue

                if can_move_block(block, cand, other_blocks):
                    return cand
        return None