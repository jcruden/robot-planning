# Update with velocity etc.
import pygame
import numpy as np
from exploration.astar import planner
import random

class Robot():
    #################
    def __init__(self, x, y, generated_map, lidar, grid, random=False, weighted_elevation=False, weighted_info=False):
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
        self.weighted_info = weighted_info
        self.weighted_elevation = weighted_elevation


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
    
    def find_frontier(self):
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

        if self.weighted_info:
            min_info = min(f[2] for f in frontiers)
            max_info = max(f[2] for f in frontiers)
            min_path = min(f[3] for f in frontiers)
            max_path = max(f[3] for f in frontiers)
            info_den = (max_info - min_info) if (max_info - min_info) != 0 else 1.0
            path_den = (max_path - min_path) if (max_path - min_path) != 0 else 1.0

            def score_frontier(f):
                norm_info = (f[2] - min_info) / info_den
                norm_path = (f[3] - min_path) / path_den
                return 2 * norm_info - 1 * norm_path
            
            return max(frontiers, key=score_frontier)
        
        # Return random frontier
        if self.random:
            random_frontier = random.choice(frontiers)
            return random_frontier

        if self.weighted_elevation:
            elevation_min = np.nanmin(self.generated_map.elevationmean)
            elevation_max = np.nanmax(self.generated_map.elevationmean)
            elevation_den = (elevation_max - elevation_min) if (elevation_max - elevation_min) != 0 else 1.0

            def weighted_elevation_score(f):
                u, v = f[0], f[1]
                norm_info = (f[2] - min_info) / info_den
                norm_path = (f[3] - min_path) / path_den
                elev = self.generated_map.elevationmean[v, u] - self.generated_map.elevationmean[robot_v, robot_u]
                norm_elev = (elev - elevation_min) / elevation_den
                return 5 * norm_info - 1 * norm_elev

            return max(frontiers, key=weighted_elevation_score)
        
        # Find frontier with max info
        best_frontier = max(frontiers, key=lambda f: (f[2], -f[3]))
        return best_frontier
    
    def set_path(self):
        # Find the closest frontier
        frontier = self.find_frontier()
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