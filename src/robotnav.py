'''prmtriangles_updatedsolution.py

   This is the updated PRM solution code for the 2D triangular problem
   with post-processing and alternate connections.

   Use the PRM algorithm to find a path around polygonal obstacles.

'''

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import sys
import random
import numpy as np

from map import generated_map
from map import viz
from map.improvedLidar import Lidar
from exploration import robot
from exploration.astar import planner

WIDTH, HEIGHT = 600, 600

def find_closest_frontier(robot, generated_map):
    rows, cols = generated_map.elevationmean.shape
    robot_u = int(robot.x / generated_map.resolution)
    robot_v = int(robot.y / generated_map.resolution)

    # Create mask for NaN and neighbors
    nan_mask = np.isnan(generated_map.elevationmean)
    valid_neighbors = ~nan_mask

    # Generate grid indices
    u_indices, v_indices = np.meshgrid(np.arange(cols), np.arange(rows))
    distances = np.sqrt((u_indices - robot_u)**2 + (v_indices - robot_v)**2)

    # No cells too close to the robot
    too_close_mask = distances < 5
    nan_mask[too_close_mask] = False

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
    # Return random frontier
    #random_frontier = random.choice(frontiers)
    #return random_frontier
    
    # Find closest frontier
    #closest_frontier = min(frontiers, key=lambda f: np.sqrt((f[0] - robot_u)**2 + (f[1] - robot_v)**2))
    closest_frontier = max(frontiers, key=lambda f: (f[2] / f[3]))
    return closest_frontier

def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    clock = pygame.time.Clock()
    velocity = np.array([0.04, 0.04])

    grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    gen_map = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    lidar = Lidar(grid, world_resolution=viz.resolution, noise_std=0.1)
    rob = robot.Robot(1, 1, gen_map, lidar)
    interval = 20 # lidar update every 50 ms
    last_time = pygame.time.get_ticks()
    path = None
    curr = 0
    rob.sensor_update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()
        if current_time - last_time >= interval:
            rob.sensor_update()
            last_time = current_time

            if not path:
                # Find the closest frontier
                frontier = find_closest_frontier(rob, gen_map)
                # Convert frontier to world coordinates
                fx, fy = frontier[0] * gen_map.resolution, frontier[1] * gen_map.resolution
                rob.destination = (fx, fy)

                # Use A* to plan a path
                start = (int(rob.x / gen_map.resolution), int(rob.y / gen_map.resolution))
                goal = (frontier[0], frontier[1])
                path = planner(start, goal, gen_map.elevationmean, gen_map)
            
            if curr < len(path):
                    next_step = path[curr]
                    next_x, next_y = next_step[0] * gen_map.resolution, next_step[1] * gen_map.resolution
                    rob.moveTo(next_x, next_y)
                    curr += 1
            else:
                path = None
                curr = 0

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(rob)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
