from collections import deque
from copy import deepcopy

GRID_ROWS = 4
GRID_COLS = 5

yellow_block = {'row1': 1, 'col1': 1, 'row2': 2, 'col2': 1}  # 이동 블록

# 방해 블록
gray_blocks = [
    {'row1': 2, 'col1': 2, 'row2': 2, 'col2': 2}
]

directions = [(-1,0,"up"), (1,0,"down"), (0,-1,"left"), (0,1,"right")]

# -------- 유틸 함수 --------
def get_virtual_horizontal(block):
    return {
        'row1': block['row1'],
        'col1': max(0, block['col1'] - 1),
        'row2': block['row2'],
        'col2': min(GRID_COLS - 1, block['col2'] + 1)
    }

def get_virtual_vertical(block):
    return {
        'row1': max(0, block['row1'] - 1),
        'col1': block['col1'],
        'row2': min(GRID_ROWS - 1, block['row2'] + 1),
        'col2': block['col2']
    }

def is_overlap(a, b):
    return not (
        a['col2'] < b['col1'] or a['col1'] > b['col2'] or
        a['row2'] < b['row1'] or a['row1'] > b['row2']
    )

def is_in_bounds(block):
    return (0 <= block['row1'] <= block['row2'] < GRID_ROWS and
            0 <= block['col1'] <= block['col2'] < GRID_COLS)

def build_occupancy(blocks):
    occupied = set()
    for blk in blocks:
        for r in range(blk['row1'], blk['row2']+1):
            for c in range(blk['col1'], blk['col2']+1):
                occupied.add((r, c))
    return occupied

def block_moved(block, dr, dc):
    return {'row1': block['row1'] + dr,
            'col1': block['col1'] + dc,
            'row2': block['row2'] + dr,
            'col2': block['col2'] + dc}

def blocks_to_cells(block):
    return [(r, c) for r in range(block['row1'], block['row2']+1)
                   for c in range(block['col1'], block['col2']+1)]

# -------- 재배치 경로 가능 여부 --------
def can_move_block(start_block, target_block, other_blocks):
    visited = set()
    queue = deque([start_block])
    occupied = build_occupancy(other_blocks)
    while queue:
        cur = queue.popleft()
        if (cur['row1'], cur['col1']) == (target_block['row1'], target_block['col1']):
            return True
        if (cur['row1'], cur['col1']) in visited:
            continue
        visited.add((cur['row1'], cur['col1']))
        for dr, dc, _ in directions:
            nxt = block_moved(cur, dr, dc)
            if not is_in_bounds(nxt):
                continue
            nxt_cells = blocks_to_cells(nxt)
            if any(cell in occupied for cell in nxt_cells):
                continue
            queue.append(nxt)
    return False

# -------- 재배치 위치 찾기 (이동 블록 주변 제한 반경 내 탐색) --------
def find_relocation_position(block, occupied_cells, vh, vv, other_blocks, max_distance=2):
    h = block['row2'] - block['row1'] + 1
    w = block['col2'] - block['col1'] + 1

    # 이동 블록 현재 위치 중앙 좌표 기준
    block_center_r = (vh['row1'] + vh['row2']) // 2
    block_center_c = (vv['col1'] + vv['col2']) // 2

    candidates = []
    for r in range(max(0, block_center_r - max_distance), min(GRID_ROWS - h + 1, block_center_r + max_distance + 1)):
        for c in range(max(0, block_center_c - max_distance), min(GRID_COLS - w + 1, block_center_c + max_distance + 1)):
            candidate = {'row1': r, 'col1': c, 'row2': r + h - 1, 'col2': c + w - 1}
            if is_overlap(candidate, vh) or is_overlap(candidate, vv):
                continue
            cells = blocks_to_cells(candidate)
            if any(cell in occupied_cells for cell in cells):
                continue
            if can_move_block(block, candidate, other_blocks):
                candidates.append(candidate)

    if candidates:
        candidates.sort(key=lambda x: abs(x['row1'] - block_center_r) + abs(x['col1'] - block_center_c))
        return candidates[0]
    return None

# -------- BFS: 재배치 없이 --------
def bfs_avoid_obstacles(yellow, gray_blocks):
    queue = deque()
    visited = set()
    queue.append((yellow, []))
    gray_occupy = build_occupancy(gray_blocks)
    while queue:
        block, path = queue.popleft()
        pos = (block['row1'], block['col1'])
        if pos in visited:
            continue
        visited.add(pos)
        new_path = path + [pos]
        if block['col2'] == GRID_COLS - 1:
            return new_path
        for dr, dc, _ in directions:
            nxt = block_moved(block, dr, dc)
            if not is_in_bounds(nxt):
                continue
            nxt_cells = blocks_to_cells(nxt)
            if any(cell in gray_occupy for cell in nxt_cells):
                continue
            queue.append((nxt, new_path))
    return None

# -------- BFS: 재배치 포함 --------
def bfs_with_relocation(yellow, gray_blocks):
    queue = deque()
    visited = set()
    queue.append((yellow, deepcopy(gray_blocks), 0, [], []))
    while queue:
        cur_block, curr_grays, reloc_cnt, path, reloc_log = queue.popleft()
        pos = (cur_block['row1'], cur_block['col1'])
        key = (pos, tuple((g['row1'], g['col1'], g['row2'], g['col2']) for g in curr_grays))
        if key in visited:
            continue
        visited.add(key)
        new_path = path + [pos]
        if cur_block['col2'] == GRID_COLS - 1:
            return new_path, reloc_cnt, reloc_log

        vh = get_virtual_horizontal(cur_block)
        vv = get_virtual_vertical(cur_block)

        for dr, dc, _ in directions:
            nxt_block = block_moved(cur_block, dr, dc)
            if not is_in_bounds(nxt_block):
                continue
            if not (is_overlap(nxt_block, vh) or is_overlap(nxt_block, vv)):
                continue

            occupied_cells = build_occupancy(curr_grays)
            blocked, block_idx = False, None
            for idx, g in enumerate(curr_grays):
                if is_overlap(nxt_block, g):
                    blocked, block_idx = True, idx
                    break

            if blocked:
                gray_to_move = curr_grays[block_idx]
                occ_wo_this = build_occupancy([g for i, g in enumerate(curr_grays) if i != block_idx])
                new_grays = deepcopy(curr_grays)
                new_vh = get_virtual_horizontal(nxt_block)
                new_vv = get_virtual_vertical(nxt_block)
                other_blocks = [g for i, g in enumerate(curr_grays) if i != block_idx] + [nxt_block]
                new_pos = find_relocation_position(gray_to_move, occ_wo_this, new_vh, new_vv, other_blocks, max_distance=2)
                if new_pos:
                    new_grays[block_idx] = new_pos
                    log = reloc_log + [((gray_to_move['row1'], gray_to_move['col1']), (new_pos['row1'], new_pos['col1']))]
                    queue.append((nxt_block, new_grays, reloc_cnt+1, new_path, log))
                else:
                    # 후보 위치 없으면 장애물 제거
                    remove_grays = deepcopy(curr_grays)
                    removed_block = remove_grays.pop(block_idx)
                    log = reloc_log + [((removed_block['row1'], removed_block['col1']), "REMOVED")]
                    queue.append((nxt_block, remove_grays, reloc_cnt+2, new_path, log))
            else:
                queue.append((nxt_block, deepcopy(curr_grays), reloc_cnt, new_path, reloc_log))
    return None, None, None

# ---- 실행 ----
if __name__ == "__main__":
    path_no_reloc = bfs_avoid_obstacles(yellow_block, gray_blocks)
    if path_no_reloc:
        print("방해 블록을 피하는 경로:")
        print(path_no_reloc)
        print("재배치 없이 반출 가능")
        print('reward:', 0 - 0.001 * (len(path_no_reloc) - 1))
    else:
        print("재배치 포함 경로 탐색...")
        path_reloc, reloc_count, reloc_log = bfs_with_relocation(yellow_block, gray_blocks)
        if path_reloc:
            print("최종 경로:", path_reloc)
            print("총 재배치 횟수:", reloc_count)
            print("재배치 이력:", reloc_log)
            print('reward:', 0 - 0.001 * (len(path_reloc) - 1) - reloc_count)
        else:
            print("반출 불가")
