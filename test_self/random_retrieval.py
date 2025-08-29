import numpy as np
import random


def can_move(grid, pos_r, pos_c, block_height, block_width):
    rows, cols = grid.shape
    # 경계 검사
    if pos_r < 0 or pos_c < 0 or (pos_r + block_height) > rows or (pos_c + block_width) > cols:
        return False

    # 블록이 차지하는 영역 모두 빈 칸인지 검사
    for r in range(pos_r, pos_r + block_height):
        for c in range(pos_c, pos_c + block_width):
            if grid[r][c] != 0:
                return False
    return True

def random_retrieval(grid, start_r, start_c, block_height=1, block_width=1):
    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 상, 하, 좌, 우 (row, col)
    rows, cols = grid.shape

    pos_r, pos_c = start_r, start_c
    path = [(pos_r, pos_c)]

    steps = 0
    max_steps = 100  # 무한루프 방지

    while steps < max_steps:
        action = random.choice(actions)
        next_r = pos_r + action[0]
        next_c = pos_c + action[1]

        if can_move(grid, next_r, next_c, block_height, block_width):
            pos_r, pos_c = next_r, next_c
            path.append((pos_r, pos_c))
        steps += 1

        if pos_c == cols - 1:
            break

    return path

# 사용 예
if __name__ == "__main__":
    grid = np.zeros((4,5), dtype=int)
    start_r, start_c = 0, 0
    path = random_retrieval(grid, start_r, start_c, block_height=2, block_width=1)
    print(path)
