import heapq

def heuristic(a, b):
    # 맨해튼 거리
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    open_set = [(heuristic(start, goal), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            next_pos = (current[0] + dx, current[1] + dy)
            if 0 <= next_pos[0] < rows and 0 <= next_pos[1] < cols and grid[next_pos[0]][next_pos[1]] == 0:
                new_cost = cost + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(next_pos, goal)
                    heapq.heappush(open_set, (priority, new_cost, next_pos))
                    came_from[next_pos] = current
    return None  # 경로를 찾을 수 없음

# 예시 실행
grid = [
    [0,0,1,0],
    [1,0,1,0],
    [0,0,0,0],
    [0,1,1,0]
]
start = (0,0)
goal = (3,3)
path = a_star(grid, start, goal)
print(path)
