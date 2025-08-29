from collections import deque
from copy import deepcopy

GRID_ROWS = 4
GRID_COLS = 6

yellow_block = {'row1': 1, 'col1': 1, 'row2': 2, 'col2': 1}
gray_blocks = [
    {'row1': 0, 'col1': 3, 'row2': 0, 'col2': 4},
    {'row1': 2, 'col1': 3, 'row2': 3, 'col2': 4}
]

# 불필요한 가상영역 함수 없이:
def is_overlap(a, b):
    return not (a['col2'] < b['col1'] or a['col1'] > b['col2'] or a['row2'] < b['row1'] or a['row1'] > b['row2'])

def is_in_bounds(block):
    return (0 <= block['row1'] <= block['row2'] < GRID_ROWS) and (0 <= block['col1'] <= block['col2'] < GRID_COLS)

def build_occupancy(blocks):
    occupied = set()
    for blk in blocks:
        for r in range(blk['row1'], blk['row2'] + 1):
            for c in range(blk['col1'], blk['col2'] + 1):
                occupied.add((r, c))
    return occupied

def block_moved(block, dr, dc):
    return {
        'row1': block['row1'] + dr,
        'col1': block['col1'] + dc,
        'row2': block['row2'] + dr,
        'col2': block['col2'] + dc
    }

def blocks_to_cells(block):
    return [(r, c) for r in range(block['row1'], block['row2'] + 1)
                  for c in range(block['col1'], block['col2'] + 1)]

# (1) 방해 블록 무시하고 경로만 탐색
def find_shortest_path(yellow_block):
    directions = [(-1,0,"up"), (1,0,"down"), (0,-1,"left"), (0,1,"right")]
    queue = deque()
    visited = set()
    queue.append((yellow_block, []))
    while queue:
        cur_block, path = queue.popleft()
        curr_pos = (cur_block['row1'], cur_block['col1'])
        if curr_pos in visited:
            continue
        visited.add(curr_pos)
        new_path = path + [curr_pos]
        if cur_block['col2'] == GRID_COLS - 1:
            return new_path
        for dr, dc, dname in directions:
            nxt_block = block_moved(cur_block, dr, dc)
            if not is_in_bounds(nxt_block):
                continue
            queue.append((nxt_block, new_path))
    return None

# (2) 해당 경로 위 방해블록 후처리(재배치)
def find_blockers_on_path(path, yellow_block, gray_blocks):
    blockers = set()
    for pos in path:
        block = {
            'row1': pos[0],
            'col1': pos[1],
            'row2': pos[0] + (yellow_block['row2'] - yellow_block['row1']),
            'col2': pos[1] + (yellow_block['col2'] - yellow_block['col1'])
        }
        for g_idx, gr in enumerate(gray_blocks):
            if is_overlap(block, gr):
                blockers.add(g_idx)
    return sorted(list(blockers))

def find_relocation_position(block, occupied_cells):
    h = block['row2'] - block['row1'] + 1
    w = block['col2'] - block['col1'] + 1
    for r in range(GRID_ROWS - h + 1):
        for c in range(GRID_COLS - w + 1):
            candidate = {'row1': r, 'col1': c, 'row2': r + h - 1, 'col2': c + w - 1}
            cells = blocks_to_cells(candidate)
            if not any(cell in occupied_cells for cell in cells):
                return candidate
    return None

def relocate_blockers(blockers, gray_blocks):
    reloc_log = []
    relocate_cnt = 0
    curr_grays = deepcopy(gray_blocks)
    for idx in blockers:
        gray = curr_grays[idx]
        occ_wo_this = build_occupancy([g for gi, g in enumerate(curr_grays) if gi != idx])
        relocate_pos = find_relocation_position(gray, occ_wo_this)
        if relocate_pos:
            reloc_log.append(((gray['row1'], gray['col1']), (relocate_pos['row1'], relocate_pos['col1'])))
            curr_grays[idx] = relocate_pos
        else:
            reloc_log.append(((gray['row1'], gray['col1']), "REMOVED"))
            curr_grays[idx] = None
        relocate_cnt += 1
    final_grays = [g for g in curr_grays if g is not None]
    return relocate_cnt, reloc_log, final_grays

# ---- 전체 흐름 실행 ----
path = find_shortest_path(yellow_block)
if path is None:
    print("방해 블록이 없어도 반출 경로 자체가 없음!")
else:
    print("최단 경로(방해 블록 무시):", path)
    blockers = find_blockers_on_path(path, yellow_block, gray_blocks)
    print("경로 위 방해 블록 인덱스:", blockers)
    relocate_cnt, reloc_log, final_grays = relocate_blockers(blockers, gray_blocks)
    print(f"재배치 총 횟수: {relocate_cnt}")
    print(f"재배치 내역: {reloc_log}")
    print(f"잔존 방해블록: {final_grays}")
