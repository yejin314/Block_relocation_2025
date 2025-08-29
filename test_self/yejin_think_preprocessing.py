from collections import deque
from copy import deepcopy

# --- 기본 설정 및 헬퍼 함수 ---
GRID_ROWS = 4
GRID_COLS = 5

yellow_block = {'row1': 1, 'col1': 0, 'row2': 2, 'col2': 1}
gray_blocks = [
    {'row1': 1, 'col1': 2, 'row2': 2, 'col2': 2}
]

directions = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]


# --- 가상 영역 생성 함수 (다시 추가) ---
def get_virtual_horizontal(block):
    # 블록의 위아래 한 칸씩을 포함하는 가로로 긴 가상 영역을 반환
    return {
        'row1': max(0, block['row1'] - 1),
        'col1': block['col1'],
        'row2': min(GRID_ROWS - 1, block['row2'] + 1),
        'col2': block['col2']
    }


def get_virtual_vertical(block):
    # 블록의 좌우 한 칸씩을 포함하는 세로로 긴 가상 영역을 반환
    return {
        'row1': block['row1'],
        'col1': max(0, block['col1'] - 1),
        'row2': block['row2'],
        'col2': min(GRID_COLS - 1, block['col2'] + 1)
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
        for r in range(blk['row1'], blk['row2'] + 1):
            for c in range(blk['col1'], blk['col2'] + 1):
                occupied.add((r, c))
    return occupied


def block_moved(block, dr, dc):
    return {'row1': block['row1'] + dr, 'col1': block['col1'] + dc,
            'row2': block['row2'] + dr, 'col2': block['col2'] + dc}


def blocks_to_tuple(blocks):
    return tuple(sorted((b['row1'], b['col1'], b['row2'], b['col2']) for b in blocks))


def find_relocation_path(block_to_move, space_to_clear, static_obstacles):
    queue = deque([(block_to_move, 0)])
    visited = set([(block_to_move['row1'], block_to_move['col1'])])
    occupied_by_static = build_occupancy(static_obstacles)

    while queue:
        current_block, moves = queue.popleft()
        if not is_overlap(current_block, space_to_clear):
            return current_block, moves

        for dr, dc, _ in directions:
            next_block = block_moved(current_block, dr, dc)
            if not is_in_bounds(next_block):
                continue
            if (next_block['row1'], next_block['col1']) in visited:
                continue

            next_occupied = build_occupancy([next_block])
            if not next_occupied.isdisjoint(occupied_by_static):
                continue

            visited.add((next_block['row1'], next_block['col1']))
            queue.append((next_block, moves + 1))

    return None


def solve_puzzle(yellow_start, gray_start):
    initial_state = (yellow_start, gray_start, 0, [], [])
    queue = deque([initial_state])
    visited = set([((yellow_start['row1'], yellow_start['col1']), blocks_to_tuple(gray_start))])

    while queue:
        y_block, g_blocks, cost, path, log = queue.popleft()
        new_path = path + [(y_block['row1'], y_block['col1'])]
        if y_block['col2'] == GRID_COLS - 1:
            return new_path, cost, log

        # --- 가상 영역 로직 적용 부분 ---
        vh = get_virtual_horizontal(y_block)  # 가로 가상 영역
        vv = get_virtual_vertical(y_block)  # 세로 가상 영역

        for dr, dc, move_dir in directions:
            next_y_block = block_moved(y_block, dr, dc)

            if not is_in_bounds(next_y_block):
                continue

            # 다음 이동 위치가 가상 영역 내에 있는지 확인
            if not (is_overlap(next_y_block, vh) or is_overlap(next_y_block, vv)):
                continue  # 가상 영역 밖의 움직임은 고려하지 않음
            # --- 로직 적용 끝 ---

            colliding_gray_idx = -1
            for i, g_block in enumerate(g_blocks):
                if is_overlap(next_y_block, g_block):
                    colliding_gray_idx = i
                    break

            if colliding_gray_idx == -1:
                state_key = ((next_y_block['row1'], next_y_block['col1']), blocks_to_tuple(g_blocks))
                if state_key not in visited:
                    visited.add(state_key)
                    queue.append((next_y_block, deepcopy(g_blocks), cost + 0.001, new_path, log))
            else:
                block_to_move = g_blocks[colliding_gray_idx]
                static_obstacles = [y_block] + [g for i, g in enumerate(g_blocks) if i != colliding_gray_idx]
                relocation_result = find_relocation_path(block_to_move, next_y_block, static_obstacles)

                if relocation_result:
                    new_g_pos, num_moves = relocation_result
                    next_g_blocks = deepcopy(g_blocks)
                    next_g_blocks[colliding_gray_idx] = new_g_pos
                    state_key = ((next_y_block['row1'], next_y_block['col1']), blocks_to_tuple(next_g_blocks))
                    if state_key not in visited:
                        visited.add(state_key)
                        new_log = log + [{'from': (block_to_move['row1'], block_to_move['col1']),
                                          'to': (new_g_pos['row1'], new_g_pos['col1']), 'moves': num_moves}]
                        new_cost = cost + 0.001 + (num_moves * 1.0)
                        queue.append((next_y_block, next_g_blocks, new_cost, new_path, new_log))

    return None, None, None


if __name__ == "__main__":
    final_path, total_cost, relocations = solve_puzzle(yellow_block, gray_blocks)
    if final_path:
        print("✅ 반출 가능 경로를 찾았습니다!")
        print("-" * 30)
        print(f"노란 블록 이동 경로: {final_path}")
        print(f"총 재배치 횟수: {len(relocations)} 회")
        print("재배치 상세 이력:")
        for r in relocations:
            print(f"  - 블록 {r['from']} -> {r['to']} ({r['moves']}칸 이동)")
        print("-" * 30)
        print(f"최종 비용 (Reward: {-total_cost:.4f})")
    else:
        print("❌ 반출 가능한 경로를 찾을 수 없습니다.")