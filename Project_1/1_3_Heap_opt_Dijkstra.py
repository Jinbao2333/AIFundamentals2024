import heapq
from collections import defaultdict

def heap_opt_dijkstra(n, m, edges):
    h = defaultdict(list)
    d = [float('inf')] * (n + 1)
    d[1] = 0
    vis = [False] * (n + 1)
    q = [(0, 1)]

    for u, v, w in edges:
        h[u].append((v, w))

    while q:
        distance, begin = heapq.heappop(q)
        if vis[begin]:
            continue
        vis[begin] = True
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
