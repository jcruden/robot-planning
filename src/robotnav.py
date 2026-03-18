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
    frontiers = []
    robot_u = int(robot.x / generated_map.resolution)
    robot_v = int(robot.y / generated_map.resolution)

    for v in range(rows):
        for u in range(cols):
            if abs(u - robot_u) + abs(v - robot_v) < 5:  # Skip cells too close to the robot
                continue
            if np.isnan(generated_map.elevationmean[v, u]):
                # Check if adjacent cells are free
                for dv, du in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nv, nu = v + dv, u + du
                    if generated_map._in_bounds(nu, nv) and not np.isnan(generated_map.elevationmean[nv, nu]):
                        frontiers.append((u, v))
                        break

    if not frontiers:
        return None

    # Find closest frontier to robot
    closest_frontier = min(frontiers, key=lambda f: abs(f[0] - robot_u) + abs(f[1] - robot_v))
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
    interval = 50 # lidar update every 50 ms
    last_time = pygame.time.get_ticks()
    path = None
    curr = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()
        if current_time - last_time >= interval:
            rob.sensor_update()
            last_time = current_time

            if path:
                if curr < len(path):
                    next_step = path[curr]
                    next_x, next_y = next_step[0] * gen_map.resolution, next_step[1] * gen_map.resolution
                    rob.moveTo(next_x, next_y)
                    curr += 1
                else:
                    path = None
                    curr = 0
            else:
                # Find the closest frontier
                frontier = find_closest_frontier(rob, gen_map)
                # Convert frontier to world coordinates
                fx, fy = frontier[0] * gen_map.resolution, frontier[1] * gen_map.resolution
                rob.destination = (fx, fy)

                # Use A* to plan a path
                start = (int(rob.x / gen_map.resolution), int(rob.y / gen_map.resolution))
                goal = (frontier[0], frontier[1])
                path = planner(start, goal, gen_map.elevationmean, gen_map)

        position = (rob.x, rob.y)
        position = np.clip(position, [0, 0], [viz.width_m, viz.height_m])  # Keep within bounds
        rob.x, rob.y = position

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(rob)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
