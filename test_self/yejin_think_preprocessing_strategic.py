import heapq
from collections import deque
from copy import deepcopy

# --- 기본 설정 및 헬퍼 함수 (이전과 동일) ---
GRID_ROWS = 10
GRID_COLS = 11

yellow_block = {'row1': 4, 'col1': 2, 'row2': 5, 'col2': 4}  # 이동 블록

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
    {'row1': 3, 'col1': 5, 'row2': 4, 'col2': 6}
]

directions = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]


def get_virtual_horizontal(block): return {'row1': max(0, block['row1'] - 1), 'col1': block['col1'],
                                           'row2': min(GRID_ROWS - 1, block['row2'] + 1), 'col2': block['col2']}


def get_virtual_vertical(block): return {'row1': block['row1'], 'col1': max(0, block['col1'] - 1),
                                         'row2': block['row2'], 'col2': min(GRID_COLS - 1, block['col2'] + 1)}


def is_overlap(a, b): return not (
            a['col2'] < b['col1'] or a['col1'] > b['col2'] or a['row2'] < b['row1'] or a['row1'] > b['row2'])


def is_in_bounds(block): return (
            0 <= block['row1'] <= block['row2'] < GRID_ROWS and 0 <= block['col1'] <= block['col2'] < GRID_COLS)


def build_occupancy(blocks):
    occupied = set()
    for blk in blocks:
        for r in range(blk['row1'], blk['row2'] + 1):
            for c in range(blk['col1'], blk['col2'] + 1):
                occupied.add((r, c))
    return occupied


def block_moved(block, dr, dc): return {'row1': block['row1'] + dr, 'col1': block['col1'] + dc,
                                        'row2': block['row2'] + dr, 'col2': block['col2'] + dc}


def blocks_to_tuple(blocks): return tuple(sorted((b['row1'], b['col1'], b['row2'], b['col2']) for b in blocks))


# --- *** 여기가 핵심 변경 부분 *** ---
def find_strategic_relocation(block_to_move, space_to_clear, static_obstacles):
    """단순히 가까운 곳이 아닌, 전략적으로 유리한 위치를 찾습니다."""
    queue = deque([(block_to_move, 0)])  # (현재 블록 위치, 이동 횟수)
    visited = set([(block_to_move['row1'], block_to_move['col1'])])
    occupied_by_static = build_occupancy(static_obstacles)

    candidate_locations = []

    while queue:
        current_block, moves = queue.popleft()

        # 일단 길을 비켜주는 위치이면 후보에 추가
        if not is_overlap(current_block, space_to_clear):
            # 전략 점수 계산: 출구(오른쪽)에서 멀고, 벽(위/아래)에 가까울수록 좋다고 가정
            strategic_score = (GRID_COLS - 1 - current_block['col2']) + min(current_block['row1'],
                                                                            GRID_ROWS - 1 - current_block['row2'])
            # 최종 점수: 이동 비용은 낮을수록, 전략 점수는 높을수록 좋음
            final_score = moves - strategic_score * 0.5  # 전략 점수에 가중치 부여
            candidate_locations.append((final_score, current_block, moves))

        # 계속 탐색 (일정 깊이 이상은 탐색하지 않아 너무 느려지는 것을 방지)
        if moves > 5: continue

        for dr, dc, _ in directions:
            next_block = block_moved(current_block, dr, dc)
            if not is_in_bounds(next_block) or (next_block['row1'], next_block['col1']) in visited:
                continue
            if not build_occupancy([next_block]).isdisjoint(occupied_by_static):
                continue
            visited.add((next_block['row1'], next_block['col1']))
            queue.append((next_block, moves + 1))

    if not candidate_locations:
        return None

    # 최종 점수가 가장 낮은(가장 유리한) 후보를 선택
    candidate_locations.sort(key=lambda x: x[0])
    best_candidate = candidate_locations[0]
    return best_candidate[1], best_candidate[2]  # (최적 위치, 이동 횟수)


def solve_puzzle_unified(yellow_start, gray_start):
    counter = 0
    pq = [(0, counter, yellow_start, gray_start, [], [])]
    visited = set()

    while pq:
        cost, _, y_block, g_blocks, path, log = heapq.heappop(pq)
        state_key = ((y_block['row1'], y_block['col1']), blocks_to_tuple(g_blocks))
        if state_key in visited: continue
        visited.add(state_key)

        new_path = path + [(y_block['row1'], y_block['col1'])]
        if y_block['col2'] == GRID_COLS - 1: return new_path, cost, log

        vh, vv = get_virtual_horizontal(y_block), get_virtual_vertical(y_block)
        for dr, dc, move_dir in directions:
            next_y_block = block_moved(y_block, dr, dc)
            if not is_in_bounds(next_y_block) or not (
                    is_overlap(next_y_block, vh) or is_overlap(next_y_block, vv)): continue

            colliding_gray_idx = -1
            for i, g_block in enumerate(g_blocks):
                if is_overlap(next_y_block, g_block):
                    colliding_gray_idx = i;
                    break

            if colliding_gray_idx == -1:
                new_cost = cost + 0.001
                counter += 1
                heapq.heappush(pq, (new_cost, counter, next_y_block, deepcopy(g_blocks), new_path, log))
            else:
                block_to_move = g_blocks[colliding_gray_idx]
                static_obstacles = [y_block] + [g for i, g in enumerate(g_blocks) if i != colliding_gray_idx]

                # --- *** find_relocation_path 대신 새로운 함수 호출 *** ---
                relocation_result = find_strategic_relocation(block_to_move, next_y_block, static_obstacles)
                if relocation_result:
                    new_g_pos, num_moves = relocation_result
                    next_g_blocks_reloc = deepcopy(g_blocks);
                    next_g_blocks_reloc[colliding_gray_idx] = new_g_pos
                    reloc_cost = cost + 0.001 + 1.0
                    reloc_log = log + [{'type': 'relocate', 'from': (block_to_move['row1'], block_to_move['col1']),
                                        'to': (new_g_pos['row1'], new_g_pos['col1']), 'cost': 1.0}]
                    counter += 1
                    heapq.heappush(pq, (reloc_cost, counter, next_y_block, next_g_blocks_reloc, new_path, reloc_log))

                next_g_blocks_removed = [g for i, g in enumerate(g_blocks) if i != colliding_gray_idx]
                removal_cost = cost + 0.001 + 2.0
                removal_log = log + [
                    {'type': 'remove', 'block_at': (block_to_move['row1'], block_to_move['col1']), 'cost': 2.0}]
                counter += 1
                heapq.heappush(pq, (removal_cost, counter, next_y_block, next_g_blocks_removed, new_path, removal_log))

    return None, None, None


# --- 메인 실행 로직 ---
if __name__ == "__main__":
    print("--- 최적 비용 경로 탐색 (전략적 재배치 적용) ---")
    path, cost, event_log = solve_puzzle_unified(yellow_block, gray_blocks)

    if path:
        print("✅ 최적의 반출 경로를 찾았습니다!")
        print("-" * 30)
        print(f"노란 블록 이동 경로: {path}")
        print(f"총 이벤트 횟수: {len(event_log)} 회")
        print("상세 이력:")
        for r in event_log:
            if r['type'] == 'relocate':
                print(f"  - [재배치] 블록 {r['from']} -> {r['to']} (비용 +{r['cost']})")
            elif r['type'] == 'remove':
                print(f"  - [제거] 블록 {r['block_at']} (비용 +{r['cost']})")
        print("-" * 30)
        print(f"최종 비용 (Reward: {-cost:.4f})")
    else:
        print("❌ 반출 가능한 경로를 찾을 수 없습니다.")