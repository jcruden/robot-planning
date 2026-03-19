# Update with velocity etc.
import pygame
import numpy as np
from exploration.astar import planner
import random

class Robot():
    #################
    def __init__(self, x, y, generated_map, lidar, grid, random=False):
        # Define
        self.x = x
        self.y = y
        self.generated_map = generated_map
        self.lidar = lidar
        self.last_scan = None
        self.destination = None
        self.path = None
        self.curr = 0
        self.random = random
        self.fuel = 10
        self.grid = grid # ground truth map for fuel

    def sensor_update(self):
        if self.lidar is None or self.generated_map is None:
            return

        scan = self.lidar.scan(self.x, self.y)
        self.last_scan = scan
        self.generated_map.updateelevation(self.x, self.y, scan.elevations, scan.hit_points)
    
    def moveTo(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, vx, vy):
        self.x += vx
        self.y += vy
    
    def find_closest_frontier(self):
        rows, cols = self.generated_map.elevationmean.shape
        robot_u = int(self.x / self.generated_map.resolution)
        robot_v = int(self.y / self.generated_map.resolution)

        # Create mask for NaN and neighbors
        nan_mask = np.isnan(self.generated_map.elevationmean)
        valid_neighbors = ~nan_mask

        # Generate grid indices
        u_indices, v_indices = np.meshgrid(np.arange(cols), np.arange(rows))
        distances = np.sqrt((u_indices - robot_u)**2 + (v_indices - robot_v)**2)

        # No cells too close to the robot
        #too_close_mask = distances < 10
        #nan_mask[too_close_mask] = False

        # Find frontiers
        frontiers = []
        for dv, du in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted_nan_mask = np.roll(np.roll(nan_mask, dv, axis=0), du, axis=1)
            frontiers_mask = shifted_nan_mask & valid_neighbors
            #frontiers.extend(zip(u_indices[frontiers_mask], v_indices[frontiers_mask]))

            for u, v in zip(u_indices[frontiers_mask], v_indices[frontiers_mask]):
                # Calculate information gain as the number of unknown neighbors
                info_gain = 0
                for ddv, ddu in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nu, nv = u + ddu, v + ddv
                    if 0 <= nu < cols and 0 <= nv < rows and nan_mask[nv, nu]:
                        info_gain += 1

                frontiers.append((u, v, info_gain, distances[v, u]))

        if not frontiers:
            return None

        frontiers = [f for f in frontiers if f[3] > 5]
        
        # Return random frontier
        if self.random:
            random_frontier = random.choice(frontiers)
            return random_frontier
        
        # Find closest frontier
        #closest_frontier = min(frontiers, key=lambda f: np.sqrt((f[0] - robot_u)**2 + (f[1] - robot_v)**2))
        #closest_frontier = max(frontiers, key=lambda f: (f[2] / (1 + f[3])))
        closest_frontier = max(frontiers, key=lambda f: (f[2], -f[3]))
        return closest_frontier
    
    def set_path(self):
        # Find the closest frontier
        frontier = self.find_closest_frontier()
        # Convert frontier to world coordinates
        fx, fy = frontier[0] * self.generated_map.resolution, frontier[1] * self.generated_map.resolution
        self.destination = (fx, fy)

        # Use A* to plan a path
        start = (int(self.x / self.generated_map.resolution), int(self.y / self.generated_map.resolution))
        goal = (frontier[0], frontier[1])
        self.path = planner(start, goal, self.generated_map.elevationmean, self.generated_map)

    def move_path(self):
        if not self.path:
             self.set_path()
        if self.curr < len(self.path):
            next_step = self.path[self.curr]
            next_x, next_y = next_step[0] * self.generated_map.resolution, next_step[1] * self.generated_map.resolution
            self.fuel -= max(0, (self.grid[next_step[1], next_step[0]] - self.grid[int(self.y / self.generated_map.resolution), int(self.x / self.generated_map.resolution)]))
            self.moveTo(next_x, next_y)
            self.curr += 1
        else:
            self.path = None
            self.curr = 0
    
    def calculate_score(self):
        explored = np.sum(~np.isnan(self.generated_map.elevationmean))
        total = self.generated_map.elevationmean.size
        coverage = explored / total
        rmse = np.sqrt(np.nanmean((self.generated_map.elevationmean - self.grid)**2))
        normalized_rmse = rmse / (np.nanstd(self.generated_map.elevationmean))
        return coverage * (1 - normalized_rmse), normalized_rmse