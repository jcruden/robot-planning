import pygame
import numpy as np
import random
from exploration.astar import planner

class VoxelRobot():
    
    def __init__(self, x, y, voxelmap, lidar, grid, random=False):
        self.x = x
        self.y = y
        self.voxelmap = voxelmap
        self.lidar = lidar
        self.grid = grid
        self.random = random
        self.fuel = 15
        self.curr = 0
        self.destination = None
        self.path = None
    
    def sensor_update(self):
        scan = self.lidar.scan(self.x,self.y)
        self.voxelmap.scan_update(scan)
        return self.voxelmap.logodds
        
    def get_elevation(self):
        cols, rows = self.grid.shape
        col = self.x / 0.05
        row = self.y / 0.05

        c0 = int(np.floor(col))
        r0 = int(np.floor(row))
        c1 = c0 + 1
        r1 = r0 + 1

        if c0 < 0 or r0 < 0 or c1 >= cols or r1 >= rows:
            return 0.0

        dc = col - c0
        dr = row - r0

        z = (self.grid[r0, c0] * (1 - dc) * (1 - dr) +
            self.grid[r0, c1] * dc * (1 - dr) +
            self.grid[r1, c0] * (1 - dc) * dr +
            self.grid[r1, c1] * dc * dr)

        return float(z)
    
    def moveTo(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, vx, vy):
        self.x += vx
        self.y += vy
    
    def find_closest_frontier(self):
        elevationmap = self.voxelmap.get_heightmap()
        rows, cols = elevationmap.shape
        robot_u = int(self.x / self.voxelmap.resolution)
        robot_v = int(self.y / self.voxelmap.resolution)

        # Create mask for NaN and neighbors
        nan_mask = np.isnan(elevationmap)
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
        fx, fy = frontier[0] * self.voxelmap.resolution, frontier[1] * self.voxelmap.resolution
        self.destination = (fx, fy)

        # Use A* to plan a path
        start = (int(self.x / self.voxelmap.resolution), int(self.y / self.voxelmap.resolution))
        goal = (frontier[0], frontier[1])
        self.path = planner(start, goal, self.voxelmap.get_heightmap(), self.voxelmap)

    def move_path(self):
        if not self.path:
             self.set_path()
        if self.curr < len(self.path):
            next_step = self.path[self.curr]
            next_x, next_y = next_step[0] * self.voxelmap.resolution, next_step[1] * self.voxelmap.resolution
            self.fuel -= max(0, (self.grid[next_step[1], next_step[0]] - self.grid[int(self.y / self.voxelmap.resolution), int(self.x / self.voxelmap.resolution)]))
            self.moveTo(next_x, next_y)
            self.curr += 1
        else:
            self.path = None
            self.curr = 0
    
    def calculate_score(self):
        elevations = self.voxelmap.get_heightmap()
        explored = np.sum(~np.isnan(elevations))
        total = elevations.size
        coverage = explored / total
        rmse = np.sqrt(np.nanmean((np.repeat(np.repeat(elevations, 2, axis=0), 2, axis=1)[:-1,:-1] - self.grid)**2))
        normalized_rmse = rmse / (np.nanstd(elevations))
        return coverage * (1 - rmse), normalized_rmse    