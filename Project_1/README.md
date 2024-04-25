

|**课程名称**：AI 基础|**年级**：2022 级|**上机实践日期**：2024 年 4 月 12 日|
|:----|:----|:----|
|**指导教师**：杨彬|**姓名**：姜嘉祺|    |
|**上机实践名称**：实验 1|**学号**：10225501447|    |


---
# 一、实验任务

## **实验目的与前景介绍：** 

本实验结合了教材第三章内容，即讨论当问题求解不能通过单个行动一步完成时，Agent 如何找到一组行动序列达到目标。搜索即是指从问题出发寻找解的过程。

本实验聚焦于几种经典的算法，包括了 BFS, DFS, Dijkstra, $A*$ 等；旨在通过实际求解问题，理解运用经典算法在实际当中的应用。


---
## **实验任务：**

* **Task 0：热身练习**
  - 完成农民、狼、羊、白菜过河问题的状态空间图。
  - 完成课堂上介绍的机场问题的求解。
* **Task 1：图最短路径问题**
\- 应用并实现不同的搜索算法来寻找图中两点之间的最短路径。

1. **BFS 算法**，在所有边权重均为 1 时找到最短路径；
2. **朴素版 Dijkstra 算法**，用于解决带有非负权重的图中最短路径问题；
3. **堆优化版 Dijkstra 算法**，通过优先队列优化提高原始 Dijkstra 算法的效率，将面对更多更严格的测试样例。
* **Task 2：八数码问题**
1. 用 **DFS（深度优先搜索）** 检查目标状态是否存在解；
2. 用 **BFS（广度优先搜索）** 找出从起始状态到目标状态的最少移动步数；
3. 用 **Dijkstra 算法**（一种特殊形式的 A*）处理上一个问题；
4. 使用 $A*$**搜索算法**，结合启发式函数计算启发式代价以寻找最少步数解决方案。
* Task 3：迷宫问题
分别采用 **DFS**、**BFS**、**Dijkstra**以及 $A*$**算法**寻找从起点到终点的最优路径，并且补充已有代码，实现对已搜索过的格子染色问题的可视化代码编写。


# 二、使用环境

使用环境为 **Python 3**。


---
# 三、实验过程

## Task 0：热身练习

### 过河问题状态转移图

对于过河问题，我们可以在每个状态时考虑下一步可能的状态，并逐个列举。其中可能会进入一些不合法的状态，例如狼和羊在一起、羊和菜在一起等；对于这些状态，我们无法进行下一步的状态转换，也无法返回上一步的状态。为此在图中我们用红色框进行展示，并将具体不合法之处标红。对于最后的目标状态，我们使用绿色框进行展示。对应的箭头或双向箭头代表可以从一个状态转移到下一个状态，或转移回来。灰色曲线箭头与黑色直线箭头行使相同的功能，作此区分仅为了防止箭头之间互相交错而引起阅读困难与歧义。

下面是使用流程图展示的状态转移图。

!['过河问题状态转移图'](https://raw.githubusercontent.com/Jinbao2333/AIFundamentals2024/5118eefad429aa6627b7f52802a06971888c836f/Project_1/FlowChart.svg)

### 机场问题

由题，我们首先可以刻画出问题的状态空间，三个机场的坐标为变量构成即六维空间. 假设用 $C_i$ 表示（当前状态下）离机场 $i$ 最近的城市集合，那么当前状态的邻接状态中，各 $C_i$ 保持常量，我们有：

 $$f(x_1,y_1,x_2,y_2,x_3,y_3)=\sum_{i=1}^{3}{\sum_{c\in C_i}{(x_i-x_c)^2+(y_i-y_c)^2}}$$ 

同时我们对上述问题计算梯度，则有：

 $$\nabla f=(\frac{\partial f}{\partial x_1},\frac{\partial f}{\partial y_1},\frac{\partial f}{\partial x_2},\frac{\partial f}{\partial y_2},\frac{\partial f}{\partial x_3},\frac{\partial f}{\partial y_3})$$ 

我们可以发现， $$\frac{\partial f}{\partial x_1}=2\sum_{c\in C_1}(x_1-x_c)$$ 

在一些简单的情形之下，如只有一个机场需要求解时，我们通过令该式为 $0$ 可以求出解. 但是对于这个问题，我们会发现找不到一个闭合解. 想要找到 $f$ 的最值，我们只能通过找到梯度为 $0$ 的 $\vec{x}$ ，也即求解 $\nabla f(\vec x)=0$ .

我们可以通过下述公式更新当前状态来完成最陡上升爬山法：

 $$\vec x=\vec x -H^{-1}_{f}(\vec x)\nabla f(\vec x)$$ 

其中 $H_{f}(\vec x)$ 是二阶导数的黑塞矩阵，其中矩阵中各元素有如下定义 $$H_{ij}=\frac{\partial ^2 f}{\partial x_i \partial x_j}$$ 通过计算可以求得：

 $$ \nabla f(\vec x)=\begin{pmatrix}2\sum_{c\in C_1}(x_1-x_c)\\2\sum_{c\in C_1}(y_1-y_c)\\2\sum_{c\in C_2}(x_2-x_c) \\2\sum_{c\in C_2}(y_2-y_c) \\2\sum_{c\in C_3}(x_3-x_c) \\2\sum_{c\in C_3}(y_3-y_c) \\\end{pmatrix} $$ . 

 $$H_{f}(\vec x)=diag(2\sum_{c\in C_1}(1),2\sum_{c\in C_1}(1),2\sum_{c\in C_2}(1),2\sum_{c\in C_2}(1),2\sum_{c\in C_3}(1),2\sum_{c\in C_3}(1))$$ . 

 $$H_f^{-1}(\vec x)=diag(\ \cfrac{1}{2\sum_{c\in C_1}(1)},\\cfrac{1}{2\sum_{c\in C_1}(1)},\\cfrac{1}{2\sum_{c\in C_2}(1)},\\cfrac{1}{2\sum_{c\in C_2}(1)},\\cfrac{1}{2\sum_{c\in C_3}(1)},\\cfrac{1}{2\sum_{c\in C_3}(1)}\ )$$ . 

由此我们可以得到 $$ \vec x=\vec x -H^{-1}_{f}(\vec x)\nabla f(\vec x)=\begin{pmatrix}\cfrac{\sum_{c\in C_1}(x_c)}{\sum_{c\in C_1}}\\ \cfrac{\sum_{c\in C_1}(y_c)}{\sum_{c\in C_1}}\\ \cfrac{\sum_{c\in C_2}(x_c)}{\sum_{c\in C_2}}\\ \cfrac{\sum_{c\in C_2}(y_c)}{\sum_{c\in C_2}}\\ \cfrac{\sum_{c\in C_3}(x_c)}{\sum_{c\in C_3}}\\ \cfrac{\sum_{c\in C_3}(y_c)}{\sum_{c\in C_3}}\\ \end{pmatrix} $$ . 

## Task 1：图最短路径问题

#### **BFS 算法求解最短路**

这道题使用 BFS 算法求解，属于基本题，实现了一个求解无向（可能有环）图中从节点 1 到节点 n 的最短路径长度的算法，采用了 BFS 的方式并使用一个双端队列来存储待处理的节点，以及数组来记录节点的访问状态和距离。

具体实现方式已在如下代码注释中给出。

```python
from collections import deque

def shortest_path(n, edges):
    graph = [[] for _ in range(n + 1)]
    for a, b in edges:
        graph[a].append(b)

    # 存储待处理的节点
    queue = deque([1])
    visited = [False] * (n + 1) 
    # 起始节点
    visited[1] = True
    distance = [float('inf')] * (n + 1)
    # 起始节点到自身的距离为 0
    distance[1] = 0

    # BFS
    while queue:
        # 取出当前节点
        current = queue.popleft()
        for neighbor in graph[current]:
            # 如果没访问过就记录下来
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True
                distance[neighbor] = distance[current] + 1

    if distance[n] != float('inf'):
        return distance[n]
    else:
        return -1

n, m = map(int, input().split())
edges = [list(map(int, input().split())) for _ in range(m)]

print(shortest_path(n, edges))
```

#### 朴素版 Dijkstra 求解最短路

对于有权重的图，我们可以使用 Dijkstra 算法进行求解，首先是朴素版 Dijkstra 算法，未经过任何优化，用于求解无向图中从节点 1 到节点 n 的最短路径长度。我们通过遍历每个节点来选择当前距离起点最近的未访问节点，并进行松弛操作，更新最短距离。尽管朴素 Dijkstra 算法没有使用堆优化，而是采用了线性搜索的方式来找到当前距离起点最近的节点，因此在节点数量较多的情况下可能效率较低。当然对应地，测试样例的数量和要求也将放松一些。

具体实现方式已在如下代码注释中给出。

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
        if min_idx == -1:  # 如果没有符合的节点就 break
            break
        vis[min_idx] = True  # 将找到的节点标记为已访问
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

#### 堆优化版 Dijkstra

上一个朴素 Dijkstra 算法在选择当前距离起点最近的节点时，采用了最为简单的线性搜索的方式，而堆优化 Dijkstra 则使用了优先队列来高效地找到当前最近的节点。在堆优化 Dijkstra 中，我们通过将节点按照到起点的距离进行排序，每次从优先队列中取出距离起点最近的节点，大大提高了算法的效率。

具体实现方式已在如下代码注释中给出。

```python
import heapq
from collections import defaultdict

def heap_opt_dijkstra(n, m, edges):
    h = defaultdict(list)  # defaultdict，用于存储每个节点的邻接节点及对应的边的权重
    d = [float('inf')] * (n + 1)  # 距离列表 d，存储从起点到每个节点的最短距离
    d[1] = 0
    vis = [False] * (n + 1)
    q = [(0, 1)]  # 优先队列 q，存储节点到起点的距离及节点编号

    for u, v, w in edges:
        h[u].append((v, w))

    while q:
        distance, begin = heapq.heappop(q)  # 从优先队列中取出距离起点最近的节点
        if vis[begin]:  # 如果当前节点已被访问过则跳过
            continue
        vis[begin] = True  # 将当前节点标记为已访问
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


---
## Task 2：八数码问题

#### DFS 求八数码问题解存在性

第一步是判断八数码问题是否有解，在这里我们使用的算法是 DFS。具体地在 DFS 过程中，对于每个状态，我们首先检查其是否达到了目标状态，如果是，则返回 True 表示可解；否则，将当前状态加入已访问状态集，并尝试向四个方向移动来生成新状态。如果新状态未被访问过，则将其加入栈中等待进一步搜索……最终，如果执行完DFS却仍未找到目标状态，则返回False表示不可解。

具体实现方式已在如下代码中给出。

```python
def is_solvable(board):
    # 一二维数组相互转换
    board_1d = board.split()
    board_1d = [int(x) if x != 'x' else 0 for x in board_1d]
    board_2d = [board_1d[i:i+3] for i in range(0, len(board_1d), 3)]

    target_state = [1, 2, 3, 4, 5, 6, 7, 8, 0]

    def dfs(current_state):
        stack = [(current_state, set())]  # 使用栈进行 DFS
        
        while stack:
            current_state, visited = stack.pop()  # 弹出当前状态和已访问状态集合
            if current_state == target_state:  # 如果当前状态等于目标状态，即有解
                return True
            
            visited.add(tuple(current_state))  # 将当前状态转换为元组并加入已访问状态集合
            blank_index = current_state.index(0)  # 获取空格索引并转换为二维数组的行列坐标
            row, col = divmod(blank_index, 3)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            
            for dr, dc in directions:  # 遍历四个方向
                new_row, new_col = row + dr, col + dc  # 新的行列坐标
                if 0 <= new_row < 3 and 0 <= new_col < 3:  # 新坐标是否合法
                    new_state = current_state[:]
                    new_blank_index = new_row * 3 + new_col  # 新的空格索引
                    # 将空格与相邻的数字交换位置
                    new_state[blank_index], new_state[new_blank_index] = new_state[new_blank_index], new_state[blank_index]
                    if tuple(new_state) not in visited:  # 若新状态未被访问过
                        stack.append((new_state, visited))  # 将新状态和已访问状态集合加入栈中
        return False  # 如果栈为空仍未找到目标状态，则不可解
    
    initial_state = board_2d
    
    return dfs(sum(initial_state, []))

input_board = input()
if is_solvable(input_board):
    print("1")
else:
    print("0")
```

#### BFS 求解最小移动步数

求解八数码问题最小移动步数需要注意的点在于我们必须先判断有无解，否则可能输出错误。有别于先前一道题，这里还需考虑后续计算移动步数的开销，故我们考虑通过八数码问题有解的充要条件，即八数码棋盘顺序列出的序列为偶排序，来判断有解与否。首先，通过 `count_inversions()` 函数计算逆序数，以判断初始状态是否可解——若逆序数为偶数则有解；然后，使用bfs函数对八数码问题进行搜索，每次从队列中取出当前状态及步数，若当前状态等于目标状态则返回步数，否则将当前状态的可行移动加入队列并继续搜索，直到队列为空或者找到目标状态为止。

具体实现方式已在如下代码中给出。

```python
from collections import deque

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
    blank_i, blank_j = divmod(blank_index, 3)  # 将空格位置转换为二维坐标
    for move in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        new_i, new_j = blank_i + move[0], blank_j + move[1]
        if 0 <= new_i < 3 and 0 <= new_j < 3:
            new_blank_index = new_i * 3 + new_j
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]  # 移动操作
            moves.append(new_board)  # 将新状态加入移动列表
    return moves

def bfs(board, target_state):
    queue = deque([(board, 0)])
    visited = set()
    while queue:
        current_board, steps = queue.popleft()  # 取出队列中的当前状态和步数
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回步数
            return steps
        visited.add(tuple(current_board))  # 将当前状态转换为元组并加入已访问集合
        for move in generate_moves(current_board):  # 遍历当前状态的所有可行移动
            if tuple(move) not in visited:  # 如果新状态未被访问过
                queue.append((move, steps + 1))  # 将新状态和步数加入队列
                visited.add(tuple(move))  # 将新状态加入已访问集合
    return -1

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = bfs(input_board, target_state)
    if steps != -1:
        print(steps)
    else:
        print("-1000")  # 输出 -1000 用作调试，一般不可能出现
else:
    print("-1")
```

#### Dijkstra 求解最小移动步数

与前一题类似，但此题要求使用 Dijkstra 算法来解决。首先仍然是判断有无解，操作一样；然后，使用 Dijkstra 算法进行搜索，使用优先队列来维护当前状态的搜索顺序，每次从堆中取出步数最小的状态，若当前状态等于目标状态则返回步数，否则将当前状态的可行移动加入堆并继续搜索，直到堆为空或者找到目标状态为止。

具体实现方式可见如下代码。

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
            new_blank_index = new_i * 3 + new_j  # 转换为一维索引
            new_board = board[:]
            new_board[blank_index], new_board[new_blank_index] = new_board[new_blank_index], new_board[blank_index]
            moves.append(new_board)  # 将新状态加入移动列表
    return moves  # 返回所有可行移动的列表

def dijkstra(board, target_state):
    heap = [(0, board)]  # 初始化堆，存储步数和当前状态
    visited = set()
    while heap:
        steps, current_board = heapq.heappop(heap)  # 弹出堆中的当前状态和步数
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回步数
            return steps
        if tuple(current_board) not in visited:  # 如果当前状态未被访问过
            visited.add(tuple(current_board))  # 将当前状态加入已访问集合
            for move in generate_moves(current_board):  # 遍历当前状态的所有可行移动
                heapq.heappush(heap, (steps + 1, move))  # 将新状态和步数加入堆中
    return -1

target_state = ['1', '2', '3', '4', '5', '6', '7', '8', 'x']
input_board = input().split()
if count_inversions(input_board) % 2 == 0:
    steps = dijkstra(input_board, target_state)
    if steps != -1:
        print(steps)
    else:
        print("-1000") # 调试用
else:
    print("-1")
```

#### A* 求解最少移动详细步骤

我们已经了解到，当我们使用该算法时，通常会使用一个评估函数用来估计从起始节点到目标节点的最佳路径长度，其通常由两部分组成：

* $g(n)$：表示从起始节点到节点 n 的实际代价，即已经花费的路径长度；
* $h(n)$：表示从节点 n 到目标节点的估计代价，即预计还需花费的路径长度。
在 $A*$ 算法中，每个节点 n 的优先级由以下公式确定：

$$f(n) = g(n) + h(n)$$

对此，我们可以使用这样的思想来解决这个问题求解八数码问题的详细移动步骤。首先判断有无解，若有解则使用 `a_star()` 函数对八数码问题进行 $A*$ 搜索。在搜索过程中，利用一个开放集来存储待探索的节点，并使用优先队列维护节点的搜索顺序。每次从开放集中取出估计代价最小的状态进行探索，若当前状态等于目标状态就返回移动步骤的序列，否则将当前状态的可行移动加入开放集，并根据实际代价（已走步数）和启发式代价（估计剩余步数）计算新状态的估计代价，以更新优先级。最终进行相应输出。

具体实现方式可见如下代码。

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
            moves.append((new_board, dir_name))  # 将新状态和移动方向加入移动列表
    return moves

def cost_estimate(board, target_state):
    count = 0
    for i in range(len(board)):
        if board[i] != target_state[i]:  # 如果当前状态与目标状态不相同，代价加一
            count += 1
    return count

def a_star(board, target_state):
    open_set = [(0 + cost_estimate(board, target_state), board, "")]  # 初始化开放集，存储估计代价、当前状态和移动序列
    heapq.heapify(open_set)
    closed_set = set()  # 初始化闭合集
    while open_set:
        f, current_board, moves_so_far = heapq.heappop(open_set)  # 从开放集中 pop 估计代价最小的状态
        if current_board == target_state:  # 如果当前状态等于目标状态，则返回移动序列
            return moves_so_far
        if tuple(current_board) not in closed_set:  # 如果当前状态未在闭合集中
            closed_set.add(tuple(current_board))  # 将当前状态加入闭合集
            for next_board, move_dir in generate_moves(current_board):  # 遍历当前状态的所有可行移动
                g = len(moves_so_far) + 1  # 计算新的移动序列长度
                h = cost_estimate(next_board, target_state)  # 计算新状态到目标状态的代价
                f = g + h  # 计算新状态的估计代价
                heapq.heappush(open_set, (f, next_board, moves_so_far + move_dir))  # 将新状态加入开放集
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

---
## Task 3：迷宫问题

第三部分的迷宫问题要求我们使用之前所用的四种算法进行解决，并且要补充可视化代码，最终要进行课堂展示。可视化代码由于大部分已给出，且非这门课的重点，故在报告中略去实现细节，只关注于算法部分。

#### BFS 求解迷宫问题

首先是 BFS 算法，从起点开始，逐层向外扩展搜索，这就保证了找到的路径是最短路径。

具体的实现细节如下代码所示。

```python
from collections import deque
import matplotlib.pyplot as plt

def MazeBFS(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = []

    parent = {}

    queue = deque([(0, 0)])  # 使用双端队列作为 BFS 的队列，初始将起点加入队列

    while queue:
        row, col = queue.popleft()  # 弹出队首

        if row == n - 1 and col == m - 1:  # 判断是否到达终点
            path = []
            while (row, col) != (0, 0):  # 通过父节点信息回溯路径
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, visited  # 返回路径长度、路径以及探索过的格子

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 判断新格子是否在迷宫范围内 and 没有被访问过 and 可达
            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in visited:
                visited.append((new_row, new_col))  # 标记新格子为已访问
                parent[(new_row, new_col)] = (row, col)  # 记录新格子的父节点信息
                queue.append((new_row, new_col))  # 将新格子加入队列，准备继续探索

    return -1, [], visited

def visualize_maze_with_path(maze, path, explored_cells):
    plt.figure(figsize=(len(maze[0]), len(maze)))
    plt.imshow(maze, cmap='Greys', interpolation='nearest')

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, marker='o', markersize=8, color='red', linewidth=3)

    max_alpha = 0.8
    min_alpha = 0.2
    alpha_step = (max_alpha - min_alpha) / len(explored_cells)

    current_alpha = max_alpha
    for cell in explored_cells:
        plt.fill([cell[1]-0.5, cell[1] + 0.5, cell[1] + 0.5, cell[1]-0.5],
                 [cell[0]-0.5, cell[0]-0.5, cell[0] + 0.5, cell[0] + 0.5],
                 color='blue', alpha=current_alpha)
        current_alpha -= alpha_step

    plt.xticks(range(len(maze[0])))
    plt.yticks(range(len(maze)))
    plt.gca().set_xticks([x - 0.5 for x in range(1, len(maze[0]))], minor=True)
    plt.gca().set_yticks([y - 0.5 for y in range(1, len(maze))], minor=True)
    plt.grid(which="minor", color="black", linestyle='-', linewidth=2)

    plt.axis('on')
    plt.show()

n, m = map(int, input().split())
maze = [list(map(int, input().split())) for _ in range(n)]

result, path, explored_cells = MazeBFS(n, m, maze)
print("Minimum number of moves:", result)
print("Path:", path)
print("Explored cells:", explored_cells)
visualize_maze_with_path(maze, path, explored_cells)
```

#### DFS 求解迷宫问题

相较于 BFS 算法本身的特性，DFS 一条路走到死的特点注定了找到最短路径相对比较麻烦。换句话说，即便是到达了终点，也无法保证中间走过的是最短路径，于是我们必须添加必要的手段来确保找到了最短路径而非仅仅是到达了终点。具体地，我们递归地探索迷宫中的每个可通行的格子，使用栈来记录待探索的下一个格子，并使用字典记录每个格子的父节点，以便在找到终点后重构路径。

具体的实现细节如下代码所示。

```python
import matplotlib.pyplot as plt

def MazeDFS(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    stack = [(0, 0)]
    # 父节点
    parent = {(0, 0): None}
    explored_cells = []

    while stack:
        # 出栈一个格子进行探索
        row, col = stack.pop()
        explored_cells.append((row, col))

        if row == n - 1 and col == m - 1:
            # 重构路径
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return len(path) - 1, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m and maze[new_row][new_col] == 0 and (new_row, new_col) not in parent:
                # 记录新格子的父节点
                parent[(new_row, new_col)] = (row, col)
                # 将新格子加入栈中
                stack.append((new_row, new_col))

    return -1, [], explored_cells

# 此后同上，省略 . . .
```

#### Dijkstra 求解迷宫问题

Dijkstra 算法，与先前的 BFS 和 DFS 相比，在搜索路径时可以考虑权重问题。它通过维护一个优先队列来选择当前距离起点最近的格子进行探索，不断更新每个格子到起点的最短距离，并记录下已经探索过的格子和路径。相比之下，BFS 和 DFS 只考虑了格子之间的相对位置关系，没有考虑路径的权重，因此无法保证找到的路径是最短的。

核心算法部分的代码实现如下。

```python
import heapq
import matplotlib.pyplot as plt

def MazeDijkstra(n, m, maze):

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 字典 dist 用于存储每个格子到起点的最短距离
    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0

    # 优先队列(距离, (行坐标, 列坐标))
    pq = [(0, (0, 0))]

    explored_cells = [(0, 0)]

    parent = {}

    # Dijkstra 循环
    while pq:
        # 从队列中取出当前距离最短的格子
        d, (row, col) = heapq.heappop(pq)

        # 如果当前格子为终点，则……
        if row == n - 1 and col == m - 1:
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return d, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            # 如果新格子位置合法
            if 0 <= new_row < n and 0 <= new_col < m:
                # 纳入
                new_dist = d + 1

                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist, (new_row, new_col)))
                    explored_cells.append((new_row, new_col))
                    parent[(new_row, new_col)] = (row, col)

    return -1, [], set()

# 此后同 3-1，省略 . . .
```

#### A* 求解迷宫问题

这个算法与 BFS 和 DFS 不同之处在于它综合了实际代价和启发式函数，以寻找最短路径。算法利用优先队列维护了一个待探索的单元格集合，并根据当前的实际代价和启发式函数的值来选择下一个要探索的单元格，以此来达到高效地搜索最短路径的目的。

具体实现如下。

```python
import heapq
import matplotlib.pyplot as plt

def heuristic(start, goal):
    # 计算曼哈顿距离作为启发式函数
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

def MazeAstar(n, m, maze):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    dist = {(i, j): float('inf') for i in range(n) for j in range(m)}
    dist[(0, 0)] = 0

    # 优先队列 (f = g + h, cell)，存储待访问的单元格及其估计代价
    pq = [(0 + heuristic((0, 0), (m - 1, n - 1)), (0, 0))]

    explored_cells = [(0, 0)]

    parent = {}

    while pq:
        # 从优先队列中取出估计代价最小的单元格
        f, (row, col) = heapq.heappop(pq)
        g = f - heuristic((row, col), (m - 1, n - 1))  # 计算实际代价

        if row == n - 1 and col == m - 1:
            # 回溯路径
            path = []
            while (row, col) != (0, 0):
                path.append((row, col))
                row, col = parent[(row, col)]
            path.append((0, 0))
            path.reverse()
            return g, path, explored_cells

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc

            if 0 <= new_row < n and 0 <= new_col < m:
                new_dist = g + 1  # 计算新的距离

                if new_dist < dist[(new_row, new_col)] and maze[new_row][new_col] == 0:
                    # 更新距离、优先队列、探索过的单元格和父节点信息
                    dist[(new_row, new_col)] = new_dist
                    heapq.heappush(pq, (new_dist + heuristic((new_row, new_col), (m - 1, n - 1)), (new_row, new_col)))
                    explored_cells.append((new_row, new_col))
                    parent[(new_row, new_col)] = (row, col)

    return -1, [], set()

# 此后同 4-1，省略 . . .
```

---
# 四、总结

这次实验通过在具体的问题中运用四大经典搜索算法，包括BFS、DFS、DIjkstra和 $A*$ ，使得我们在具体问题中感受经典算法的优劣之处，并对相应问题有了更明确的算法选择。


---
## 实验收获：

虽然看似这个实验是关于数据结构与算法的，还没有涉及到真正的AI内容，但是这个实验仍然有着重要的意义。诚然，学习BFS、DFS、Dijkstra和 $A*$ 算法是必要的，正是因为这些算法是人工智能领域中常用的搜索和路径规划方法，能够帮助我们理解问题空间的结构、搜索最优解的策略以及如何在计算机中实现这些策略。只有通过亲自实践学习这些算法，才能帮助我们掌握应用它们解决各种实际问题的方法，为进一步深入学习和应用AI奠定基础。

