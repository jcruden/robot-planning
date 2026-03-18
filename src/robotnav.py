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

WIDTH, HEIGHT = 600, 900

def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    clock = pygame.time.Clock()
    velocity = np.array([0.04, 0.04])

    grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    gen_map = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    gen_map2 = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    lidar = Lidar(grid, world_resolution=viz.resolution, noise_std=0.1)
    rob = robot.Robot(1, 1, gen_map, lidar)
    lidar2 = Lidar(grid, world_resolution=viz.resolution, noise_std=0.1)
    rob2 = robot.Robot(1, 1, gen_map2, lidar2, random=True)
    interval = 20 # lidar update every 50 ms
    last_time = pygame.time.get_ticks()
    path = None
    curr = 0
    rob.sensor_update()
    rob2.sensor_update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()
        if current_time - last_time >= interval:
            rob.sensor_update()
            rob2.sensor_update()
            last_time = current_time

            rob.move_path()
            rob2.move_path()

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(rob, rob2)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
