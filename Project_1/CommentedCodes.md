## 1-1

```python
from collections import deque

def bfs_shortest_path(graph, start, end):
    visited = set()
    queue = deque([(start, 0)])
    
    while queue:
        node, distance = queue.popleft()
        if node == end:
            return distance
        visited.add(node)  # 将当前节点标记为已访问
        for neighbor in graph[node]:  # 遍历当前节点的邻居节点
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))  # 将邻居节点及其到达距离加入队列，距离加一
    
    return -1

n, m = map(int, input().split())
graph = {i: [] for i in range(1, n+1)}  # 初始化图邻接表

for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

print(bfs_shortest_path(graph, 1, n))
```

## 1-2
```python
def naive_dijkstra(n, m, edges):
    # 堆 h 存储每个节点的邻接节点及对应的边权重
    h = [[] for _ in range(n + 1)]
    # 距离列表 d 存储从起点到每个节点的最短距离
    d = [float('inf')] * (n + 1)
    d[1] = 0
    # 访问标记列表 vis
    vis = [False] * (n + 1)

    # 邻接表，将每条边的信息添加到堆中
    for u, v, w in edges:
        h[u].append((v, w))

    # 循环，每次选择一个未访问节点进行松弛操作
    for _ in range(n):
        min_dist = float('inf')
        min_idx = -1
        # 遍历所有节点，找到未访问节点中距离起点最近的节点
        for i in range(1, n + 1):
            if not vis[i] and d[i] < min_dist:
                min_dist = d[i]
                min_idx = i
        if min_idx == -1:  # 如果没有符合的节点就 break
            break
        vis[min_idx] = True  # 将找到的节点标记为已访问
        # 对于选定节点的所有邻接节点，进行松弛操作，更新最短距离
        for v, w in h[min_idx]:
            if d[v] > d[min_idx] + w:
                d[v] = d[min_idx] + w

    if d[n] != float('inf'):
        return d[n]
    return -1

n, m = map(int, input().split())
edges = []
for _ in range(m):
    x, y, z = map(int, input().split())
    edges.append((x, y, z))

print(naive_dijkstra(n, m, edges))
```

## 1-3
```python
import heapq
from collections import defaultdict

def heap_opt_dijkstra(n, m, edges):
    h = defaultdict(list)  # defaultdict，用于存储每个节点的邻接节点及对应的边的权重
    d = [float('inf')] * (n + 1)  # 距离列表 d，存储从起点到每个节点的最短距离
    d[1] = 0
    vis = [False] * (n + 1)
    q = [(0, 1)]  # 优先队列 q，存储节点到起点的距离及节点编号

    for u, v, w in edges:
        h[u].append((v, w))

    while q:
        distance, begin = heapq.heappop(q)  # 从优先队列中取出距离起点最近的节点
        if vis[begin]:  # 如果当前节点已被访问过则跳过
            continue
        vis[begin] = True  # 将当前节点标记为已访问
        # 对于当前节点的所有邻接节点，进行松弛操作，更新最短距离，并将新的距离加入优先队列
        for j, w in h[begin]:
            if d[j] > d[begin] + w:
                d[j] = d[begin] + w
                heapq.heappush(q, (d[j], j))

    if d[n] != float('inf'):
        return d[n]
    return -1

n, m = map(int, input().split())
edges = []
for _ in range(m):
    x, y, z = map(int, input().split())
    edges.append((x, y, z))

print(heap_opt_dijkstra(n, m, edges))
```

## 2-1
```python
def is_solvable(board):
    board_1d = board.split()
    board_1d = [int(x) if x != 'x' else 0 for x in board_1d]
    board_2d = [board_1d[i:i+3] for i in range(0, len(board_1d), 3)]

    target_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

    def dfs(current_state):
        stack = [(current_state, set())]  # 使用栈进行 DFS
        
        while stack:
            current_state, visited = stack.pop()  # 弹出当前状态和已访问状态集合
            if current_state == target_state:  # 如果当前状态等于目标状态，即表示可解
                return True
            
            visited.add(tuple(current_state))  # 将当前状态转换为元组并加入已访问状态集合
            blank_index = current_state.index(0)  # 获取空格索引并转换为二维数组的行列坐标
            row, col = divmod(blank_index, 3)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四个方向
            
            for dr, dc in directions:  # 遍历四个方向
                new_row, new_col = row + dr, col + dc  # 新的行列坐标
                if 0 <= new_row < 3 and 0 <= new_col < 3:  # 新坐标是否合法
                    new_state = current_state[:]
                    new_blank_index = new_row * 3 + new_col  # 新的空格索引
                    # 将空格与相邻的数字交换位置
                    new_state[blank_index], new_state[new_blank_index] = new_state[new_blank_index], new_state[blank_index]
                    if tuple(new_state) not in visited:  # 若新状态未被访问过
                        stack.append((new_state, visited))  # 将新状态和已访问状态集合加入栈中
        return False  # 如果栈为空仍未找到目标状态，即不可解
    
    initial_state = board_2d
    
    return dfs(sum(initial_state, []))  # 调用函数并将二维数组转换为一维数组作为参数

input_board = input()
if is_solvable(input_board):
    print("1")
else:
    print("0")
```

## 2-2
```python
from collections import deque
import numpy as np

def count_inversions(board):
    inversion_count = 0
    board_1d = [x for x in board if x != 'x']
    for i in range(len(board_1d)):
        for j in range(i + 1, len(board_1d)):
            if board_1d[i] > board_1d[j]:
                inversion_count += 1
    return inversion_count

def generate_moves(board):
    moves = []
    for i in range(len(board)):
        if board[i] == 'x':
            blank_index = i
    blank_i, blank_j = divmod(blank_index, 3)  # 将空格位置转换为二维坐标
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]  # 移动操作
            moves.append(new_board)  # 将新状态加入移动列表
    return moves

def bfs(board, target_state):
    queue = deque([(board, 0)])
    visited = set()
    while queue:
        current_board, steps = queue.popleft()  # 取出队列中的当前状态和步数
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回步数
            return steps
        visited.add(tuple(current_board))  # 将当前状态转换为元组并加入已访问集合
        for move in generate_moves(current_board):  # 遍历当前状态的所有可行移动
            if tuple(move) not in visited:  # 如果新状态未被访问过
                queue.append((move, steps + 1))  # 将新状态和步数加入队列
                visited.add(tuple(move))  # 将新状态加入已访问集合
    return -1

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = bfs(input_board, target_state)
    if steps != -1:
        print(steps)
    else:
        print("-1000")  # 输出 -1000 用作调试，一般不可能出现
else:
    print("-1")
```

## 2-3
```python
import heapq

def count_inversions(board):
    inversion_count = 0
    board_1d = [x for x in board if x != 'x']
    for i in range(len(board_1d)):
        for j in range(i + 1, len(board_1d)):
            if board_1d[i] > board_1d[j]:
                inversion_count += 1
    return inversion_count

def generate_moves(board):
    moves = []
    for i in range(len(board)):
        if board[i] == 'x':
            blank_index = i
    blank_i, blank_j = divmod(blank_index, 3)
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j  # 转换为一维索引
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]
            moves.append(new_board)  # 将新状态加入移动列表
    return moves  # 返回所有可行移动的列表

def dijkstra(board, target_state):
    heap = [(0, board)]  # 初始化堆，存储步数和当前状态
    visited = set()
    while heap:
        steps, current_board = heapq.heappop(heap)  # 弹出堆中的当前状态和步数
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回步数
            return steps
        if tuple(current_board) not in visited:  # 如果当前状态未被访问过
            visited.add(tuple(current_board))  # 将当前状态加入已访问集合
            for move in generate_moves(current_board):  # 遍历当前状态的所有可行移动
                heapq.heappush(heap, (steps + 1, move))  # 将新状态和步数加入堆中
    return -1

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = dijkstra(input_board, target_state)
    if steps != -1:
        print(steps)
    else:
        print("-1000")
else:
    print("-1")
```

## 2-4
```python
import heapq

def count_inversions(board):
    inversion_count = 0
    board_1d = [x for x in board if x != 'x']
    for i in range(len(board_1d)):
        for j in range(i + 1, len(board_1d)):
            if board_1d[i] > board_1d[j]:
                inversion_count += 1
    return inversion_count

def generate_moves(board):
    moves = []
    for i in range(len(board)):
        if board[i] == 'x':
            blank_index = i
    blank_i, blank_j = divmod(blank_index, 3)
    for move, dir_name in [((0, 1), 'r'), ((0, -1), 'l'), ((1, 0), 'd'), ((-1, 0), 'u')]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]
            moves.append((new_board, dir_name))  # 将新状态和移动方向加入移动列表
    return moves

def cost_estimate(board, target_state):
    count = 0
    for i in range(len(board)):
        if board[i] != target_state[i]:  # 如果当前状态与目标状态不相同，代价加一
            count += 1
    return count

def a_star(board, target_state):
    open_set = [(0 + cost_estimate(board, target_state), board, "")]  # 初始化开放集，存储估计代价、当前状态和移动序列
    heapq.heapify(open_set)
    closed_set = set()  # 初始化闭合集
    while open_set:
        f, current_board, moves_so_far = heapq.heappop(open_set)  # 从开放集中 pop 估计代价最小的状态
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回移动序列
            return moves_so_far
        if tuple(current_board) not in closed_set:  # 如果当前状态未在闭合集中
            closed_set.add(tuple(current_board))  # 将当前状态加入闭合集
            for next_board, move_dir in generate_moves(current_board):  # 遍历当前状态的所有可行移动
                g = len(moves_so_far) + 1  # 计算新的移动序列长度
                h = cost_estimate(next_board, target_state)  # 计算新状态到目标状态的代价
                f = g + h  # 计算新状态的估计代价
                heapq.heappush(open_set, (f, next_board, moves_so_far + move_dir))  # 将新状态加入开放集
    return "unsolvable"

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = a_star(input_board, target_state)
    if steps != "unsolvable":
        print(steps)
    else:
        print("--error")
else:
    print("unsolvable")
```