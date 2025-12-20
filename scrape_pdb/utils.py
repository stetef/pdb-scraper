#!/usr/bin/env python3
"""General utilities."""

import math


def _slice(line: str, a: int, b: int) -> str:
    if len(line) < 80:
        line = line + (" " * (80 - len(line)))
    return line[a:b]


def dist(a: tuple[float,float,float], b: tuple[float,float,float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def centroid(points: list[tuple[float,float,float]]) -> tuple[float,float,float]:
    if not points:
        return (0.0, 0.0, 0.0)
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    n = float(len(points))
    return (sx/n, sy/n, sz/n)

def connected_components(points: list[tuple[float,float,float]], cutoff: float) -> list[list[int]]:
    """Connected components where nodes within cutoff are connected (union-of-spheres on metals)."""
    n = len(points)
    if n == 0:
        return []
    c2 = cutoff * cutoff
    adj = [[] for _ in range(n)]
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i+1, n):
            xj, yj, zj = points[j]
            if (xi-xj)**2 + (yi-yj)**2 + (zi-zj)**2 <= c2:
                adj[i].append(j)
                adj[j].append(i)
    visited = [False]*n
    comps: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    return comps