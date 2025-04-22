"""
A* planner to plan path on a given grid map.
"""

import networkx

class AstarPlanner:
    """
    A* planner to plan path on a given map.
    """
    def __init__(self, cur_map):

        # Initialize current map (fully unobserved at start)
        self.cur_map = cur_map

        # Create NetworkX graph representation of the map
        self.G = networkx.grid_2d_graph(*cur_map.shape)

        # Add diagonal edges
        for i in range(cur_map.shape[0] - 1):
            for j in range(cur_map.shape[1] - 1):
                self.G.add_edge((i, j), (i + 1, j + 1))
                self.G.add_edge((i + 1, j), (i, j + 1))

        for edge in self.G.edges():
            self.G.edges[edge]['weight'] = 1
        
        self.update_weights()

    def plan_path(self, start, goal):
        try:
            if start == goal:
                return [start, goal]
            path = networkx.shortest_path(self.G, start, goal, weight='weight')
            return path
        except (networkx.NetworkXNoPath, ValueError):
            # print("No path found to goal!")
            return None
    
    def update_map(self, new_map):
        self.cur_map = new_map
        self.update_weights()

    def update_weights(self):
        rm_edges = []
        for edge in self.G.edges():
            x1, y1 = edge[0]
            x2, y2 = edge[1]
            if self.cur_map[x1, y1] == 1 or self.cur_map[x2, y2] == 1:
                # remove edge if any of the nodes is an obstacle
                rm_edges.append(edge)
        self.G.remove_edges_from(rm_edges)
    
    def update_obstacle(self, i, j):
        cur_node = (i, j)
        rm_edges = []
        for edge in self.G.edges(cur_node):
            rm_edges.append(edge)
        self.G.remove_edges_from(rm_edges)
    
    def plan_waypoints(self, waypoints):
        path = []
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            goal = waypoints[i + 1]
            cur_path = self.plan_path(start, goal)
            if cur_path is None:
                return path
            path += cur_path
        return path
