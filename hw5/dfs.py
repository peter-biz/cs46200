
# The same thing as HW5, but using DFS instead of BFS

from collections import deque
from typing import List, Tuple

def maxMinutes(grid):
    m, n = len(grid), len(grid[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    def bfs_fire():
        fire_times = [[float('inf')] * n for _ in range(m)]
        fire_queue = deque()
        
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1:
                    fire_queue.append((x, y))
                    fire_times[x][y] = 0
        
        while fire_queue:
            x, y = fire_queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 0 and fire_times[nx][ny] == float('inf'):
                    fire_times[nx][ny] = fire_times[x][y] + 1
                    fire_queue.append((nx, ny))
        
        return fire_times
    
    def dfs_player(x, y, wait_time, visited):
        if x == m - 1 and y == n - 1:
            return True
        
        visited[x][y] = True
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 0 and not visited[nx][ny]:
                player_move_time = wait_time + 1
                fire_time = fire_times[nx][ny]
                if player_move_time < fire_time or (nx == m-1 and ny == n-1 and player_move_time <= fire_time):
                    if dfs_player(nx, ny, player_move_time, visited):
                        return True
        
        visited[x][y] = False
        return False
    
    fire_times = bfs_fire()
    left, right = 0, 10**9
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        visited = [[False] * n for _ in range(m)]
        if dfs_player(0, 0, mid, visited):
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

# Test cases
test_cases = [
    [[0,2,0,0,0,0,0],[0,0,0,2,2,1,0],[0,2,0,0,1,2,0],[0,0,2,2,2,0,2],[0,0,0,0,0,0,0]],
    [[0,0,0,0],[0,1,2,0],[0,2,0,0]],
    [[0,0,0],[2,2,0],[1,2,0]]
]

for grid in test_cases:
    print(f"Grid: {grid}")
    print(f"Maximum minutes: {maxMinutes(grid)}\n")