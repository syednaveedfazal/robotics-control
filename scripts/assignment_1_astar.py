#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from collections import deque # Necessary for efficient FIFO queue
import heapq
def plot_path(path, n_start, n_goal, M, save_fp=None):
    plt.matshow(M, cmap="gray")
    if path.shape[0] > 2:
        plt.plot(path[:, 1], path[:, 0], 'b')
    plt.plot(n_start[1], n_start[0], 'or')
    plt.plot(n_goal[1], n_goal[0], 'xg')
    if save_fp is None:
        plt.show()
    else:
        plt.savefig(save_fp)
    plt.close()


def is_valid(v, thr_free=0.9):
    """
    v : freespace value of the map
    thr_free : threshold of a cell on the map to be considered free
    """
    if v > thr_free:
        return True
    return False


class Node:
    """Use this class to help with keeping track of the paths"""
    def __init__(self, parent=None, cell=None):
        self.parent = parent
        self.cell = cell 
        self.g = 0
        self.h = 0
        self.f = 0
        
    def __str__(self):
        return str(self.cell)
    
    def __repr__(self):
        return str(self)


def plan_path_uninformed(n_start, n_goal, M):
    """
    Exe 1, Q1. Implement BFS (Uniform Cost with unit cost) here.
    """
    
    # Initialize visited map to keep track of where we have been
    # Using the same shape as M. 0 = unvisited, 1 = visited.
    visited = np.zeros(M.shape)
    
    # Initialize the queue (FIFO)
    queue = deque()
    
    # Create the start node
    start_node = Node(parent=None, cell=n_start)
    start_node.g = 0
    
    # Add start to queue and mark as visited
    queue.append(start_node)
    visited[n_start[0], n_start[1]] = 1
    
    goal_node_found = None
    
    # Define possible movements: Up, Down, Left, Right
    # Assuming (row, col) coordinates
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    path = np.array([]) # Default to empty

    while len(queue) > 0:
        
        # 1. Pick one point (FIFO) to explore and remove it from queue
        current_node = queue.popleft()
        current_pos = current_node.cell
        
        # 2. Check if the point is n_goal
        # Compare element-wise
        if current_pos[0] == n_goal[0] and current_pos[1] == n_goal[1]:
            goal_node_found = current_node
            break
        
        # 3. Get neighbours and check their validity
        for move in movements:
            new_x = current_pos[0] + move[0]
            new_y = current_pos[1] + move[1]
            
            # Check grid boundaries
            if 0 <= new_x < M.shape[0] and 0 <= new_y < M.shape[1]:
                
                # Check if cell is valid (free space) AND not visited
                if is_valid(M[new_x, new_y]) and visited[new_x, new_y] == 0:
                    
                    # 4. Add neighbours to queue
                    new_node = Node(parent=current_node, cell=np.array([new_x, new_y]))
                    new_node.g = current_node.g + 1 # Cost is path length
                    
                    queue.append(new_node)
                    visited[new_x, new_y] = 1 # Mark as visited immediately when adding to queue
        
    # Reconstruct path if goal was found
    if goal_node_found is not None:
        path_list = []
        curr = goal_node_found
        while curr is not None:
            path_list.append(curr.cell)
            curr = curr.parent
        # Reverse the list to go from Start -> Goal
        path = np.array(path_list[::-1])
        
    return path, visited

# Placeholders for other functions to prevent errors in test script
def plan_path_astar(n_start, n_goal, M):
    return [], np.zeros(M.shape)

def plan_path_fast(n_start, n_goal, M):
    return [], np.zeros(M.shape)



# --- Add this Heuristic Function ---
def heuristic(a, b):
    # Euclidean Distance
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

# --- Implement the A* Function ---
def plan_path_astar(n_start, n_goal, M):
    """
    Exe 2, Q2. Implement A* here
    """
    # Grid dimensions
    rows, cols = M.shape
    
    # Priority Queue (Min-Heap) for Open List
    # Stores tuples: (f_score, cell_coordinates)
    # We store coordinates instead of Node objects to avoid comparison issues in the heap
    open_list = []
    
    # Data structures to keep track of costs and parents
    # Initialize g_score with infinity
    g_score = np.full((rows, cols), np.inf)
    g_score[n_start[0], n_start[1]] = 0
    
    # Parent pointers to reconstruct path: parent[x][y] = (parent_x, parent_y)
    # We use a dictionary for sparse storage or a 3D array. Dictionary is easier here.
    came_from = {}
    
    # Calculate initial f_score
    start_h = heuristic(n_start, n_goal)
    start_f = start_h # g is 0
    
    # Push start node
    heapq.heappush(open_list, (start_f, tuple(n_start)))
    
    # Visited set (Closed List) to ensure we don't process nodes redundantly
    visited = np.zeros(M.shape)
    
    # Possible movements (4-connected)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

    path = np.array([]) # Default empty

    while open_list:
        # 1. Pop node with lowest f_score
        current_f, current_cell = heapq.heappop(open_list)
        current_x, current_y = current_cell
        
        # If we already found a shorter path to this node in closed list, skip
        # (Standard optimization in A*)
        if visited[current_x, current_y] == 1:
            continue

        # Mark as visited
        visited[current_x, current_y] = 1
        
        # 2. Check Goal
        if current_x == n_goal[0] and current_y == n_goal[1]:
            # Reconstruct path
            path_list = []
            curr = current_cell
            while curr in came_from:
                path_list.append(curr)
                curr = came_from[curr]
            path_list.append(tuple(n_start)) # Add start
            path = np.array(path_list[::-1]) # Reverse
            return path, visited

        # 3. Expand Neighbors
        for dx, dy in movements:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            
            # Check bounds and obstacle
            if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                # Assuming simple is_valid check (M[x,y] > 0.9 is free)
                # If your map uses 0 as free, swap logic. 
                # Based on previous context (mpimg), usually 1 is white (free).
                if M[neighbor_x, neighbor_y] > 0.9: 
                    
                    tentative_g = g_score[current_x, current_y] + 1
                    
                    # If this path to neighbor is better than any previous one
                    if tentative_g < g_score[neighbor_x, neighbor_y]:
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                        g_score[neighbor_x, neighbor_y] = tentative_g
                        h = heuristic((neighbor_x, neighbor_y), n_goal)
                        f = tentative_g + h
                        
                        heapq.heappush(open_list, (f, (neighbor_x, neighbor_y)))

    return path, visited


def heuristic(a, b):
    # Euclidean Distance
    return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

def plan_path_fast(n_start, n_goal, M):
    """
    Exe 3, Q2. Weighted A* Implementation
    Analysis: 
    - Uses a weight > 1 applied to the heuristic.
    - This biases the search to move towards the goal much more aggressively.
    - Trade-off: Much faster computation, slightly longer path.
    """
    # Grid dimensions
    rows, cols = M.shape
    
    # Priority Queue
    open_list = []
    
    # Heuristic Weight (The "Aggressiveness" factor)
    # w = 1.0 is standard A* (Optimal)
    # w > 1.0 is Weighted A* (Faster, Suboptimal)
    # w = 3.0 is usually a good balance for speed in grids
    weight = 3.0 
    
    # Cost to reach current node
    g_score = np.full((rows, cols), np.inf)
    g_score[n_start[0], n_start[1]] = 0
    
    # Path reconstruction dictionary
    came_from = {}
    
    # Initial push
    start_h = heuristic(n_start, n_goal)
    # f = g + w * h
    start_f = 0 + weight * start_h
    
    heapq.heappush(open_list, (start_f, tuple(n_start)))
    
    visited = np.zeros(M.shape)
    
    # standard 4-connected movement
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    path = np.array([]) 

    while open_list:
        # Pop lowest f_score
        current_f, current_cell = heapq.heappop(open_list)
        current_x, current_y = current_cell
        
        if visited[current_x, current_y] == 1:
            continue
        visited[current_x, current_y] = 1
        
        # Check Goal
        if current_x == n_goal[0] and current_y == n_goal[1]:
            path_list = []
            curr = current_cell
            while curr in came_from:
                path_list.append(curr)
                curr = came_from[curr]
            path_list.append(tuple(n_start))
            path = np.array(path_list[::-1])
            return path, visited

        # Expand Neighbors
        for dx, dy in movements:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            
            if 0 <= neighbor_x < rows and 0 <= neighbor_y < cols:
                # Check obstacle (assuming > 0.9 is free)
                if M[neighbor_x, neighbor_y] > 0.9:
                    
                    tentative_g = g_score[current_x, current_y] + 1
                    
                    if tentative_g < g_score[neighbor_x, neighbor_y]:
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                        g_score[neighbor_x, neighbor_y] = tentative_g
                        
                        h = heuristic((neighbor_x, neighbor_y), n_goal)
                        # Apply Weight here
                        f = tentative_g + (weight * h)
                        
                        heapq.heappush(open_list, (f, (neighbor_x, neighbor_y)))

    return path, visited