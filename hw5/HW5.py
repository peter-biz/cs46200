# Only using basic libraries (built in libraries)
# You are given a 0-indexed 2D integer array grid of size m x n which represents a field. Each cell has one of three values:
# 0 represents grass,
# 1 represents fire,
# 2 represents a wall that you and fire cannot pass through.
# You are situated in the top-left cell, (0, 0), and you want to travel to the safehouse at the bottom-right cell, (m - 1, n - 1). 
# Every minute, you may move to an adjacent grass cell. After your move, every fire cell will spread to all adjacent cells that are not walls.
# Return the maximum number of minutes that you can stay in your initial position before moving while still safely reaching the safehouse. 
# If this is impossible, return -1. If you can always reach the safehouse regardless of the minutes stayed, return 109.
# Note that even if the fire spreads to the safehouse immediately after you have reached it, it will be counted as safely reaching the safehouse.
# A cell is adjacent to another cell if the former is directly north, east, south, or west of the latter (i.e., their sides are touching).

from collections import deque

def maxMinutes(grid): 
    m = len(grid) # Number of rows
    n = len(grid[0]) # Number of columns

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    # BFS for the fire
    def bfs_fire():
        fire_times = [[float('inf')] * n for _ in range(m)] # Initialize fire times with infinity, because we don't know how long it will take for the fire to reach a cell
        fire_queue = deque() # Using deque because we need to pop from the left

        # For each cell in the grid
        for x in range(m):
            for y in range(n):
                if grid[x][y] == 1: # Check if the cell is on fire
                    fire_queue.append((x, y))
                    fire_times[x][y] = 0

        # While there are cells on fire
        while fire_queue:
            x, y = fire_queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy # Calculate the next cell
                # Check if the cell is within the grid, is grass, and has not been visited
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 0 and fire_times[nx][ny] == float('inf'):
                    fire_times[nx][ny] = fire_times[x][y] + 1
                    fire_queue.append((nx, ny))

        return fire_times
    
    # BFS for the player, using BFS instead of DFS because it's more efficient than DFS for finding
    # the shortest path in and unweighted graph
    def bfs_player(max_wait):
        visited = [[False] * n for _ in range(m)] # Initialize visited with False
        player_queue = deque([(0, 0, max_wait)])
        visited[0][0] = True # Mark the starting cell as visited

        while player_queue:
            x, y, wait_time = player_queue.popleft()
            if x == m - 1 and y == n - 1: # If the player has reached the safehouse
                return True
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy # Calculate the next cell
                # Check if the cell is within the grid, is grass, and has not been visited
                if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 0 and not visited[nx][ny]:
                    player_move_time = wait_time + 1
                    fire_time = fire_times[nx][ny]
                    # Check if the player can move to the cell without being caught by the fire
                    if player_move_time < fire_time or (nx == m-1 and ny == n-1 and player_move_time <= fire_time):
                        player_queue.append((nx, ny, player_move_time))
                        visited[nx][ny] = True
                    
        return False
    
    fire_times = bfs_fire() # Get the fire times
    left, right = 0, 10**9 # Initialize the left and right pointers
    result = -1 # Initialize the result

    while left <= right:
        mid = (left + right) // 2
        # Check if the player can reach the safehouse within the given time
        if bfs_player(mid):
            result = mid
            left = mid + 1
        else:
            right = mid - 1

    return result
    



def main():
    choice = input("Enter 1 for default test cases, 2 to enter your own test cases: ")
    if choice.strip() == '1': 
        test_cases = [
        [[0,2,0,0,0,0,0],[0,0,0,2,2,1,0],[0,2,0,0,1,2,0],[0,0,2,2,2,0,2],[0,0,0,0,0,0,0]],
        [[0,0,0,0],[0,1,2,0],[0,2,0,0]],
        [[0,0,0],[2,2,0],[1,2,0]]
    ]

        for grid in test_cases:
            print(f"Grid: {grid}")
            print(f"Maximum minutes: {maxMinutes(grid)}\n")
    else:
        m = int(input("Enter the number of rows: "))
        n = int(input("Enter the number of columns: "))
        grid = []
        for i in range(m):
            row = list(map(int, input().split())) # Enter the row elements separated by space, e.g. 0 2 0 0 0 0 0
            grid.append(row)
        print(f"Maximum minutes: {maxMinutes(grid)}")

if __name__ == "__main__":
    main()