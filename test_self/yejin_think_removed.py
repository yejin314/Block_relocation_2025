from yejin_think_preprocessing import is_overlap, build_occupancy

def find_relocation_position_for_return(block, occupied_cells, GRID_ROWS, GRID_COLS):
    h = block['row2'] - block['row1'] + 1
    w = block['col2'] - block['col1'] + 1
    for r in range(GRID_ROWS - h + 1):
        for c in range(GRID_COLS - w + 1):
            candidate = {'row1': r, 'col1': c, 'row2': r+h-1, 'col2': c+w-1}
            cells = [(row, col) for row in range(candidate['row1'], candidate['row2']+1)
                                  for col in range(candidate['col1'], candidate['col2']+1)]
            if not any(cell in occupied_cells for cell in cells):
                return candidate
    return None

def return_removed_blocks_to_grid(removed_blocks, existing_blocks, GRID_ROWS, GRID_COLS):
    curr_blocks = existing_blocks[:]
    occupied_cells = set()
    for blk in curr_blocks:
        for r in range(blk['row1'], blk['row2']+1):
            for c in range(blk['col1'], blk['col2']+1):
                occupied_cells.add((r, c))
    restored = []
    for blk in removed_blocks:
        pos = find_relocation_position_for_return(blk, occupied_cells, GRID_ROWS, GRID_COLS)
        if pos:
            restored.append(pos)
            for r in range(pos['row1'], pos['row2']+1):
                for c in range(pos['col1'], pos['col2']+1):
                    occupied_cells.add((r, c))
        else:
            # 복귀 불가 (그리드가 꽉 찬 경우 등)
            restored.append(None)
    return restored

# 적치장 반환 로직이 붙은 BFS+재배치 예시
# (반출 성공 시, 경로 로그와 함께 REMOVED 블록들을 다시 적치장에 복귀)
def bfs_with_relocation_and_return(GRID_ROWS, GRID_COLS, yellow_block, gray_blocks):
    from collections import deque
    from copy import deepcopy
    directions = [(-1,0,"up"), (1,0,"down"), (0,-1,"left"), (0,1,"right")]
    queue = deque()
    visited = set()
    queue.append((yellow_block, deepcopy(gray_blocks), 0, [], [], []))  # [removed_blocks에 누적]
    while queue:
        cur_block, curr_grays, reloc_cnt, path, reloc_log, removed_blocks = queue.popleft()
        curr_pos = (cur_block['row1'], cur_block['col1'])
        key = (curr_pos, tuple((g['row1'],g['col1'],g['row2'],g['col2']) for g in curr_grays))
        if key in visited:
            continue
        visited.add(key)
        new_path = path + [curr_pos]
        if cur_block['col2'] == GRID_COLS - 1:
            # 반출 목표 지점 도달: 삭제 블록들 복귀 시도
            restored = return_removed_blocks_to_grid(removed_blocks, [g for g in curr_grays if g is not None], GRID_ROWS, GRID_COLS)
            print("경로:", new_path)
            print("재배치 이력:", reloc_log)
            print("삭제된 블록 복귀 결과(복귀 좌표, None=복귀실패):", restored)
            return new_path, reloc_cnt, reloc_log, restored

        vh = {
            'row1': cur_block['row1'],
            'col1': max(0, cur_block['col1'] - 1),
            'row2': cur_block['row2'],
            'col2': min(GRID_COLS-1, cur_block['col2'] + 1)
        }
        vv = {
            'row1': max(0, cur_block['row1'] - 1),
            'col1': cur_block['col1'],
            'row2': min(GRID_ROWS-1, cur_block['row2'] + 1),
            'col2': cur_block['col2']
        }
        for dr, dc, dname in directions:
            nxt_block = {
                'row1': cur_block['row1'] + dr,
                'col1': cur_block['col1'] + dc,
                'row2': cur_block['row2'] + dr,
                'col2': cur_block['col2'] + dc
            }
            if not (0 <= nxt_block['row1'] <= nxt_block['row2'] < GRID_ROWS and 0 <= nxt_block['col1'] <= nxt_block['col2'] < GRID_COLS):
                continue
            # 방해 블록과 겹침 체크
            blocked, block_idx = False, None
            for idx, g in enumerate(curr_grays):
                if g is not None and is_overlap(nxt_block, g):
                    blocked, block_idx = True, idx
                    break
            if blocked:
                gray_to_move = curr_grays[block_idx]
                occ_wo_this = build_occupancy([g for i, g in enumerate(curr_grays) if i != block_idx and g is not None])
                # 가상 영역은 기존처럼
                new_vh = {
                    'row1': nxt_block['row1'],
                    'col1': max(0, nxt_block['col1'] - 1),
                    'row2': nxt_block['row2'],
                    'col2': min(GRID_COLS-1, nxt_block['col2'] + 1)
                }
                new_vv = {
                    'row1': max(0, nxt_block['row1'] - 1),
                    'col1': nxt_block['col1'],
                    'row2': min(GRID_ROWS-1, nxt_block['row2'] + 1),
                    'col2': nxt_block['col2']
                }
                new_pos = find_relocation_position_for_return(gray_to_move, occ_wo_this, GRID_ROWS, GRID_COLS)
                new_grays = deepcopy(curr_grays)
                if new_pos:
                    new_grays[block_idx] = new_pos
                    log = reloc_log + [((gray_to_move['row1'], gray_to_move['col1']), (new_pos['row1'], new_pos['col1']))]
                    queue.append((nxt_block, new_grays, reloc_cnt+1, new_path, log, removed_blocks))
                else:
                    # 복귀 불가→삭제, REMOVED 블록 추적
                    remove_grays = deepcopy(curr_grays)
                    removed_block = remove_grays[block_idx]
                    remove_grays[block_idx] = None
                    log = reloc_log + [((removed_block['row1'], removed_block['col1']), "REMOVED")]
                    queue.append((nxt_block, remove_grays, reloc_cnt+1, new_path, log, removed_blocks + [removed_block]))
            else:
                queue.append((nxt_block, deepcopy(curr_grays), reloc_cnt, new_path, reloc_log, removed_blocks))
    print("경로 없음 or 복귀 불가")
    return None, None, None, None

# 샘플 실행
GRID_ROWS = 10
GRID_COLS = 11

yellow_block = {'row1': 7, 'col1': 0, 'row2': 8, 'col2': 2}  # 이동 블록

# 방해 블록
gray_blocks = [
    # 맨 왼쪽 세로줄(위에서 아래로 연속)
    {'row1': 1, 'col1': 1, 'row2': 2, 'col2': 3},    # (0,2)-(2,2)
    # # (2,3)만 단독
    # {'row1': 2, 'col1': 3, 'row2': 2, 'col2': 3},
    # # 윗쪽 가로줄 (2행, 4~4)
    # {'row1': 2, 'col1': 4, 'row2': 2, 'col2': 4},
    # 중앙(3,4)-(4,4) 가로 연속 2칸
    {'row1': 6, 'col1': 4, 'row2': 9, 'col2': 5},
    # 세로줄 (5,3)-(6,3)
    # {'row1': 5, 'col1': 3, 'row2': 6, 'col2': 3},
    # 오른쪽 가로줄 (3,5)-(4,5)
    {'row1': 1, 'col1': 6, 'row2': 3, 'col2': 8}
]
path, reloc_count, reloc_log, restored = bfs_with_relocation_and_return(GRID_ROWS, GRID_COLS, yellow_block, gray_blocks)

print("반출 경로:", path)
print("재배치 횟수:", reloc_count)
print("재배치 이력:", reloc_log)
print("적치장 복귀 결과:", restored)
