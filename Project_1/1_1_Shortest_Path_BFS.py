from collections import deque

def bfs_shortest_path(graph, start, end):
    visited = set()
    queue = deque([(start, 0)])  # (节点, 到达该节点的距离)
    
    while queue:
        node, distance = queue.popleft()
        if node == end:
            return distance
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))

    return -1

n, m = map(int, input().split())
graph = {i: [] for i in range(1, n+1)}

for _ in range(m):
    u, v = map(int, input().split())
    graph[u].append(v)
    graph[v].append(u)

print(bfs_shortest_path(graph, 1, n))
